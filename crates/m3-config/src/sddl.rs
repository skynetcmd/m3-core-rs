//! Who a Windows DACL lets in, read from its SDDL text. A port of m3's
//! `bin/m3_core/win_acl.py` (`foreign_principals`); pure, so it is tested on
//! every OS.
//!
//! "Private" means every ALLOW ace that applies to the object names the
//! current user, SYSTEM, Administrators or OWNER RIGHTS. A local administrator
//! can read any file regardless, so admitting them adds no exposure.

/// SDDL aliases and SIDs that count as private (win_acl `_TRUSTED`).
const TRUSTED: [&str; 6] = ["SY", "BA", "OW", "S-1-5-18", "S-1-5-32-544", "S-1-3-4"];

/// The LocalSystem SID: a Windows service's account.
pub(crate) const SYSTEM_SID: &str = "S-1-5-18";

/// Readable names for the principals a too-broad ACL usually names.
fn name_of(sid: &str) -> Option<&'static str> {
    Some(match sid {
        "WD" | "S-1-1-0" => "Everyone",
        "AU" => "Authenticated Users",
        "BU" | "S-1-5-32-545" => "Users",
        "IU" => "Interactive users",
        "SU" => "Service logons",
        "S-1-5-3" => "Batch logons",
        "AN" => "Anonymous",
        _ => return None,
    })
}

/// The owner SID (`O:`) of an SDDL string, if present.
pub(crate) fn owner(sddl: &str) -> Option<&str> {
    let rest = &sddl[sddl.find("O:")? + 2..];
    let end = ["G:", "D:", "S:"]
        .iter()
        .filter_map(|m| rest.find(m))
        .min()
        .unwrap_or(rest.len());
    Some(&rest[..end]).filter(|s| !s.is_empty())
}

/// The principals an ALLOW ace grants beyond the trusted set, the current
/// user and `also_trusted`. `user` is every string that names the current user
/// in SDDL: its SID and any alias Windows writes for it instead (`LA` for the
/// built-in Administrator; `winacl::user_aliases`). A missing DACL grants
/// everyone, reported as "Everyone (no DACL)".
pub(crate) fn foreign_principals(
    sddl: &str,
    user: &[&str],
    also_trusted: Option<&str>,
) -> Vec<String> {
    let dacl = sddl.split_once("D:").map(|(_, d)| d).unwrap_or("");
    let dacl = dacl.split_once("S:").map(|(d, _)| d).unwrap_or(dacl);
    if dacl.is_empty() || dacl.starts_with("NO_ACCESS_CONTROL") {
        return vec!["Everyone (no DACL)".into()];
    }
    let mut foreign: Vec<String> = Vec::new();
    let mut rest = dacl;
    while let Some(open) = rest.find('(') {
        let after = &rest[open + 1..];
        let Some(close) = after.find(')') else { break };
        let ace = &after[..close];
        rest = &after[close + 1..];
        // win_acl's pattern matches exactly six fields without parentheses;
        // anything else (a conditional ace) is skipped, as there.
        let fields: Vec<&str> = ace.split(';').collect();
        if ace.contains('(') || fields.len() != 6 {
            continue;
        }
        let (ace_type, flags, sid) = (fields[0], fields[1], fields[5]);
        if !ace_type.starts_with('A') || flags.contains("IO") {
            continue; // a deny ace, or inherit-only
        }
        if TRUSTED.contains(&sid) || user.contains(&sid) || Some(sid) == also_trusted {
            continue;
        }
        let named = match name_of(sid) {
            Some(n) => format!("{sid} ({n})"),
            None => sid.to_string(),
        };
        if !foreign.contains(&named) {
            foreign.push(named);
        }
    }
    foreign
}

#[cfg(test)]
mod tests {
    use super::*;

    const ME: &str = "S-1-5-21-1-2-3-1001";

    #[test]
    fn a_profile_style_acl_is_private() {
        let sddl = format!("O:{ME}D:PAI(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;{ME})");
        assert!(foreign_principals(&sddl, &[ME], None).is_empty());
    }

    #[test]
    fn users_and_everyone_are_foreign_and_named() {
        let sddl = format!("O:{ME}D:(A;;FA;;;{ME})(A;;FR;;;BU)(A;ID;FR;;;WD)");
        assert_eq!(
            foreign_principals(&sddl, &[ME], None),
            vec!["BU (Users)".to_string(), "WD (Everyone)".to_string()]
        );
    }

    #[test]
    fn deny_and_inherit_only_aces_do_not_grant() {
        let sddl = format!("O:{ME}D:(A;;FA;;;{ME})(D;;FR;;;WD)(A;OICIIO;FR;;;BU)");
        assert!(foreign_principals(&sddl, &[ME], None).is_empty());
    }

    #[test]
    fn no_dacl_is_everyone() {
        assert_eq!(
            foreign_principals(&format!("O:{ME}"), &[ME], None),
            vec!["Everyone (no DACL)".to_string()]
        );
        assert_eq!(
            foreign_principals("O:SYD:NO_ACCESS_CONTROL", &[ME], None),
            vec!["Everyone (no DACL)".to_string()]
        );
    }

    #[test]
    fn the_owner_is_read_and_can_be_trusted_explicitly() {
        let sddl = format!("O:{ME}D:(A;;FA;;;SY)(A;;FA;;;{ME})");
        assert_eq!(owner(&sddl), Some(ME));
        assert_eq!(owner("O:BAD:(A;;FA;;;SY)"), Some("BA"));
        // Read as SYSTEM, the user's own file is foreign unless its owner is trusted.
        assert_eq!(
            foreign_principals(&sddl, &[SYSTEM_SID], None),
            vec![ME.to_string()]
        );
        assert!(foreign_principals(&sddl, &[SYSTEM_SID], Some(ME)).is_empty());
    }

    /// The built-in Administrator's own ace is written `LA`, never as its SID
    /// (measured on Windows 11; GitHub's Windows runner runs as that account).
    #[test]
    fn the_builtin_administrator_is_recognised_by_its_alias() {
        const ADMIN: &str = "S-1-5-21-1-2-3-500";
        let sddl = "O:LAD:PAI(A;;FA;;;LA)(A;;FA;;;SY)";
        assert!(foreign_principals(sddl, &[ADMIN, "LA"], None).is_empty());
        assert_eq!(
            foreign_principals(sddl, &[ADMIN], None),
            vec!["LA".to_string()],
            "by its SID alone, its own ace looks foreign"
        );
        assert_eq!(
            foreign_principals(sddl, &[ME], None),
            vec!["LA".to_string()],
            "for anyone else, the Administrator is not the user"
        );
    }
}
