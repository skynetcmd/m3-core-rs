//! The Windows half of the secrets-file privacy check: the current user's SID
//! and a file's owner + DACL as SDDL. The decision is [`crate::sddl`]'s, as in
//! m3's `win_acl.py`, which calls the same four advapi32 functions.

use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use std::ptr;

use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, LocalFree, HANDLE};
use windows_sys::Win32::Security::Authorization::{
    ConvertSecurityDescriptorToStringSecurityDescriptorW, ConvertSidToStringSidW,
    ConvertStringSidToSidW, GetNamedSecurityInfoW, SDDL_REVISION_1, SE_FILE_OBJECT,
};
use windows_sys::Win32::Security::{
    GetTokenInformation, TokenUser, DACL_SECURITY_INFORMATION, OWNER_SECURITY_INFORMATION,
    PSECURITY_DESCRIPTOR, PSID, TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

/// A NUL-terminated UTF-16 string from Windows, freed with LocalFree.
unsafe fn take_wide(p: *mut u16) -> String {
    let mut len = 0usize;
    while *p.add(len) != 0 {
        len += 1;
    }
    let s = String::from_utf16_lossy(std::slice::from_raw_parts(p, len));
    LocalFree(p.cast());
    s
}

/// The SID of the account this process runs as, e.g. `S-1-5-18` for a service.
pub(crate) fn current_user_sid() -> Result<String, String> {
    // SAFETY: every out-pointer is a live local; the token handle is closed on
    // every path; the TOKEN_USER buffer is sized by the first call.
    unsafe {
        let mut token: HANDLE = ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return Err(format!(
                "OpenProcessToken failed (winerror {})",
                GetLastError()
            ));
        }
        let mut size = 0u32;
        GetTokenInformation(token, TokenUser, ptr::null_mut(), 0, &mut size);
        // u64 words keep the buffer aligned for TOKEN_USER's pointer field.
        let mut buf = vec![0u64; (size as usize).div_ceil(8).max(1)];
        let ok = GetTokenInformation(token, TokenUser, buf.as_mut_ptr().cast(), size, &mut size);
        let err = GetLastError();
        CloseHandle(token);
        if ok == 0 {
            return Err(format!("GetTokenInformation failed (winerror {err})"));
        }
        let user = &*(buf.as_ptr() as *const TOKEN_USER);
        let mut out: *mut u16 = ptr::null_mut();
        if ConvertSidToStringSidW(user.User.Sid, &mut out) == 0 {
            return Err(format!(
                "ConvertSidToStringSidW failed (winerror {})",
                GetLastError()
            ));
        }
        Ok(take_wide(out))
    }
}

/// The SDDL aliases that stand for `user_sid` in a descriptor's text. SDDL
/// writes two user accounts as aliases, never as their SID: `LA` (the built-in
/// Administrator, RID 500) and `LG` (Guest, RID 501). A check that compares
/// SIDs alone reads that user's own ace as someone else's.
pub(crate) fn user_aliases(user_sid: &str) -> Vec<&'static str> {
    ["LA", "LG"]
        .into_iter()
        .filter(|a| alias_sid(a).as_deref() == Some(user_sid))
        .collect()
}

/// The SID an SDDL alias stands for on this machine, or None if Windows does
/// not resolve it.
fn alias_sid(alias: &str) -> Option<String> {
    let wide: Vec<u16> = alias.encode_utf16().chain(Some(0)).collect();
    // SAFETY: `wide` is NUL-terminated and outlives the call; the SID and the
    // string Windows allocates are freed with LocalFree.
    unsafe {
        let mut psid: PSID = ptr::null_mut();
        if ConvertStringSidToSidW(wide.as_ptr(), &mut psid) == 0 {
            return None;
        }
        let mut out: *mut u16 = ptr::null_mut();
        let ok = ConvertSidToStringSidW(psid, &mut out);
        LocalFree(psid);
        if ok == 0 {
            return None;
        }
        Some(take_wide(out))
    }
}

/// The owner and DACL of `path` as an SDDL string.
pub(crate) fn file_sddl(path: &Path) -> Result<String, String> {
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let info = OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION;
    // SAFETY: `wide` is NUL-terminated and outlives the call; the descriptor
    // and the string Windows allocates are freed with LocalFree.
    unsafe {
        let mut psd: PSECURITY_DESCRIPTOR = ptr::null_mut();
        let rc = GetNamedSecurityInfoW(
            wide.as_ptr(),
            SE_FILE_OBJECT,
            info,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut psd,
        );
        if rc != 0 {
            return Err(format!(
                "GetNamedSecurityInfoW({}) failed (winerror {rc})",
                path.display()
            ));
        }
        let mut out: *mut u16 = ptr::null_mut();
        let ok = ConvertSecurityDescriptorToStringSecurityDescriptorW(
            psd,
            SDDL_REVISION_1,
            info,
            &mut out,
            ptr::null_mut(),
        );
        let err = GetLastError();
        LocalFree(psd);
        if ok == 0 {
            return Err(format!(
                "ConvertSecurityDescriptorToStringSecurityDescriptorW failed (winerror {err})"
            ));
        }
        Ok(take_wide(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_resolves_to_this_machines_administrator_and_names_only_that_user() {
        let admin = alias_sid("LA").expect("Windows resolves LA");
        assert!(
            admin.starts_with("S-1-5-21-") && admin.ends_with("-500"),
            "{admin}"
        );
        assert_eq!(user_aliases(&admin), vec!["LA"]);
        let me = current_user_sid().unwrap();
        assert_eq!(user_aliases(&me).contains(&"LA"), me == admin, "{me}");
        assert!(user_aliases("S-1-5-18").is_empty());
    }
}
