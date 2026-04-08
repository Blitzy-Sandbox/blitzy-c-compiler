// src/frontend/preprocessor/include.rs — Include Path Resolution
//
// This module implements the include path resolution logic for `#include`
// directives in the C11 preprocessor. It resolves both angle-bracket
// (`<header.h>`) and quoted (`"header.h"`) include forms by searching
// through user-specified `-I` directories, the bundled freestanding header
// directory, and system header paths.
//
// Key responsibilities:
//   - Locate header files across multiple search directories
//   - Detect and prevent circular `#include` chains
//   - Support `#pragma once` to skip re-inclusion of guarded files
//   - Read file contents for the preprocessor to process
//   - Integrate with build.rs-provided bundled header path
//
// This module uses ONLY the Rust standard library. No external crates.
// No `unsafe` code is used anywhere in this module.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// IncludeKind — Discriminates angle-bracket vs quoted include forms
// ---------------------------------------------------------------------------

/// Specifies the kind of `#include` directive, which determines the search
/// order for locating the header file.
///
/// The two forms follow the standard C11 §6.10.2 search semantics:
///
/// - **Angle** (`<header.h>`) — Searches `-I` directories, bundled headers,
///   then system include directories. Does NOT search the current file's
///   directory.
///
/// - **Quoted** (`"header.h"`) — Searches the current file's directory first,
///   then falls through to `-I` directories, bundled headers, and system
///   include directories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncludeKind {
    /// `<header.h>` — Search system/bundled headers; skip current directory.
    Angle,
    /// `"header.h"` — Search current directory first, then -I paths, then system.
    Quoted,
}

// ---------------------------------------------------------------------------
// IncludeError — Error variants for include resolution failures
// ---------------------------------------------------------------------------

/// Errors that can occur during include path resolution, file reading,
/// or include stack management.
#[derive(Debug)]
pub enum IncludeError {
    /// Header file was not found in any search path. The `String` contains
    /// the original header name as specified in the `#include` directive.
    NotFound(String),

    /// A circular `#include` chain was detected. The `PathBuf` is the
    /// canonical path of the file that would create the cycle.
    CircularInclude(PathBuf),

    /// The file has been marked with `#pragma once` and was already
    /// included. The `PathBuf` is the canonical path of the file.
    PragmaOnce(PathBuf),

    /// An I/O error occurred while reading the file or canonicalizing
    /// a path. Wraps the underlying `io::Error`.
    IoError(io::Error),
}

impl std::fmt::Display for IncludeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IncludeError::NotFound(name) => {
                write!(f, "'{}' file not found", name)
            }
            IncludeError::CircularInclude(path) => {
                write!(f, "circular include detected: '{}'", path.display())
            }
            IncludeError::PragmaOnce(path) => {
                write!(
                    f,
                    "file already included (pragma once): '{}'",
                    path.display()
                )
            }
            IncludeError::IoError(err) => {
                write!(f, "I/O error: {}", err)
            }
        }
    }
}

/// Conversion from `io::Error` to `IncludeError::IoError` to enable
/// the `?` operator in functions that perform filesystem operations.
impl From<io::Error> for IncludeError {
    fn from(err: io::Error) -> Self {
        IncludeError::IoError(err)
    }
}

// ---------------------------------------------------------------------------
// IncludeResolver — Core include path resolution engine
// ---------------------------------------------------------------------------

/// Resolves `#include` directives to filesystem paths by searching through
/// a prioritized list of directories. Also manages the include stack for
/// circular include detection and `#pragma once` tracking.
///
/// # Search Order
///
/// **Quoted includes** (`"header.h"`):
/// 1. Current file's directory
/// 2. `-I` directories (in CLI-specified order)
/// 3. Bundled freestanding header directory
/// 4. System include directories
///
/// **Angle-bracket includes** (`<header.h>`):
/// 1. `-I` directories (in CLI-specified order)
/// 2. Bundled freestanding header directory
/// 3. System include directories
///
/// # Circular Include Detection
///
/// The resolver maintains a stack of currently-being-processed files. When
/// a new file is pushed onto the stack, it checks whether the file is
/// already on the stack (circular include) or has been `#pragma once`'d.
pub struct IncludeResolver {
    /// Directories specified via `-I` flags on the command line.
    /// Searched in the order they were specified.
    include_dirs: Vec<PathBuf>,

    /// Path to the bundled freestanding headers directory (the `include/`
    /// directory at the project root, or the path set by `build.rs` via
    /// the `BCC_BUNDLED_INCLUDE_DIR` environment variable).
    bundled_header_path: Option<PathBuf>,

    /// System header search paths (e.g., `/usr/include`, `/usr/local/include`,
    /// or cross-compilation sysroot paths for non-native targets).
    system_include_dirs: Vec<PathBuf>,

    /// Stack of canonical file paths currently being processed. The top of
    /// the stack is the file currently being preprocessed. Used for circular
    /// include detection and for determining the "current directory" for
    /// quoted include resolution.
    include_stack: Vec<PathBuf>,

    /// Set of canonical file paths that have been marked with `#pragma once`.
    /// Files in this set will be silently skipped on subsequent `#include`
    /// attempts instead of being re-processed.
    pragma_once_files: HashSet<PathBuf>,
}

impl IncludeResolver {
    /// Creates a new `IncludeResolver` with the given search directories.
    ///
    /// # Arguments
    ///
    /// * `include_dirs` — Directories from `-I` command-line flags, in the
    ///   order they were specified. These are searched after the current
    ///   directory (for quoted includes) and before bundled/system headers.
    ///
    /// * `bundled_header_path` — Optional path to the directory containing
    ///   the nine bundled freestanding C headers. If `None`, the resolver
    ///   attempts to use the compile-time path from `build.rs` via the
    ///   `BCC_BUNDLED_INCLUDE_DIR` environment variable.
    ///
    /// * `system_include_dirs` — System header directories. If empty,
    ///   no system directories are searched (the caller should provide
    ///   appropriate defaults based on the target configuration).
    pub fn new(
        include_dirs: Vec<PathBuf>,
        bundled_header_path: Option<PathBuf>,
        system_include_dirs: Vec<PathBuf>,
    ) -> Self {
        // Determine the bundled header path. If the caller did not provide
        // one, fall back to the compile-time path set by build.rs.
        let effective_bundled_path = bundled_header_path.or_else(|| {
            let compile_time_path = option_env!("BCC_BUNDLED_INCLUDE_DIR");
            compile_time_path.map(PathBuf::from).filter(|p| p.exists())
        });

        IncludeResolver {
            include_dirs,
            bundled_header_path: effective_bundled_path,
            system_include_dirs,
            include_stack: Vec::new(),
            pragma_once_files: HashSet::new(),
        }
    }

    /// Resolves an `#include` directive to a canonical filesystem path.
    ///
    /// The search order depends on the `IncludeKind`:
    ///
    /// - `Quoted`: current_dir → -I dirs → bundled → system
    /// - `Angle`:  -I dirs → bundled → system
    ///
    /// # Arguments
    ///
    /// * `name` — The header name as it appeared in the directive
    ///   (e.g., `"stdio.h"` or `"myheader.h"`).
    /// * `kind` — Whether this is an angle-bracket or quoted include.
    /// * `current_dir` — The directory of the file that contains the
    ///   `#include` directive. Used only for `Quoted` includes.
    ///
    /// # Returns
    ///
    /// The canonicalized path to the resolved header file, or an
    /// `IncludeError::NotFound` if the header could not be located
    /// in any search directory.
    pub fn resolve(
        &self,
        name: &str,
        kind: IncludeKind,
        current_dir: &Path,
    ) -> Result<PathBuf, IncludeError> {
        // For quoted includes, search the current directory first.
        if kind == IncludeKind::Quoted {
            if let Some(found) = Self::try_resolve_in_dir(current_dir, name) {
                return Ok(found);
            }
        }

        // Search -I directories in the order they were specified.
        for dir in &self.include_dirs {
            if let Some(found) = Self::try_resolve_in_dir(dir, name) {
                return Ok(found);
            }
        }

        // Search the bundled freestanding header directory.
        if let Some(ref bundled) = self.bundled_header_path {
            if let Some(found) = Self::try_resolve_in_dir(bundled, name) {
                return Ok(found);
            }
        }

        // Search system include directories.
        for dir in &self.system_include_dirs {
            if let Some(found) = Self::try_resolve_in_dir(dir, name) {
                return Ok(found);
            }
        }

        // Header not found in any search path.
        Err(IncludeError::NotFound(name.to_string()))
    }

    /// Pushes a file onto the include stack, checking for circular includes
    /// and `#pragma once` files.
    ///
    /// This method must be called before processing the contents of an
    /// included file. It canonicalizes the path and verifies that:
    ///
    /// 1. The file is not already on the include stack (circular include).
    /// 2. The file has not been marked with `#pragma once`.
    ///
    /// # Arguments
    ///
    /// * `path` — The path of the file about to be processed.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the file was successfully pushed, or an `IncludeError`
    /// if a circular include or pragma-once condition was detected.
    pub fn push_include(&mut self, path: &Path) -> Result<(), IncludeError> {
        // Canonicalize the path to resolve symlinks, `.`, and `..` so that
        // the same physical file is always represented by the same path.
        let canonical = self.canonicalize_path(path)?;

        // Check if this file has been marked with #pragma once.
        if self.pragma_once_files.contains(&canonical) {
            return Err(IncludeError::PragmaOnce(canonical));
        }

        // Check if this file is already on the include stack (circular).
        if self.include_stack.contains(&canonical) {
            return Err(IncludeError::CircularInclude(canonical));
        }

        // Push the canonical path onto the include stack.
        self.include_stack.push(canonical);
        Ok(())
    }

    /// Pops the top file from the include stack.
    ///
    /// This method must be called after the preprocessor finishes processing
    /// the contents of an included file. It restores the previous file
    /// context for subsequent include resolution.
    pub fn pop_include(&mut self) {
        self.include_stack.pop();
    }

    /// Returns the directory of the file currently being processed (the
    /// top of the include stack), or `None` if the include stack is empty.
    ///
    /// This is used as the `current_dir` argument to `resolve()` when
    /// processing `"header.h"` quoted includes, so that the search begins
    /// in the same directory as the file containing the `#include` directive.
    pub fn current_dir(&self) -> Option<&Path> {
        self.include_stack.last().and_then(|p| p.parent())
    }

    /// Marks a file as `#pragma once`, preventing future re-inclusion.
    ///
    /// The path is canonicalized before being added to the pragma-once set,
    /// ensuring that the same physical file is recognized regardless of
    /// how its path is specified in different `#include` directives.
    ///
    /// # Arguments
    ///
    /// * `path` — The path of the file to mark. Typically the file currently
    ///   being processed (top of the include stack).
    pub fn mark_pragma_once(&mut self, path: &Path) {
        // Attempt to canonicalize; if canonicalization fails (unlikely for
        // a file we're currently processing), fall back to the as-given path.
        let canonical = self
            .canonicalize_path(path)
            .unwrap_or_else(|_| path.to_path_buf());
        self.pragma_once_files.insert(canonical);
    }

    /// Checks whether a file has been marked with `#pragma once`.
    ///
    /// Returns `true` if the file was previously marked, `false` otherwise.
    /// The path is canonicalized before lookup to ensure consistent matching.
    pub fn is_pragma_once(&self, path: &Path) -> bool {
        // Attempt to canonicalize; if that fails, try the raw path as well.
        if let Ok(canonical) = self.canonicalize_path(path) {
            self.pragma_once_files.contains(&canonical)
        } else {
            self.pragma_once_files.contains(path)
        }
    }

    /// Reads the entire contents of a file as a UTF-8 string.
    ///
    /// This is a convenience method used by the preprocessor to read the
    /// contents of an included file after its path has been resolved.
    ///
    /// # Arguments
    ///
    /// * `path` — The path of the file to read.
    ///
    /// # Returns
    ///
    /// The file contents as a `String`, or an `IncludeError::IoError`
    /// if the file cannot be read.
    pub fn read_file(&self, path: &Path) -> Result<String, IncludeError> {
        fs::read_to_string(path).map_err(IncludeError::IoError)
    }

    // -----------------------------------------------------------------------
    // Private helper methods
    // -----------------------------------------------------------------------

    /// Attempts to resolve a header name within a single directory.
    ///
    /// Constructs `dir/name` and checks whether the resulting path points
    /// to an existing regular file. Returns the canonicalized path if found,
    /// or `None` if the file does not exist.
    ///
    /// This method also performs basic path traversal mitigation by verifying
    /// that the resolved file actually exists as a file (not a directory)
    /// via `fs::metadata()`.
    fn try_resolve_in_dir(dir: &Path, name: &str) -> Option<PathBuf> {
        let candidate = dir.join(name);

        // Use fs::metadata() to check existence and verify it's a file.
        // This avoids following dangling symlinks and correctly identifies
        // directories vs. files.
        match fs::metadata(&candidate) {
            Ok(meta) if meta.is_file() => {
                // Attempt to canonicalize the path for consistent matching
                // across the include stack and pragma_once set. If
                // canonicalization fails, return the joined path as-is.
                Some(candidate.canonicalize().unwrap_or(candidate))
            }
            _ => None,
        }
    }

    /// Canonicalizes a path, resolving symlinks, `.`, and `..` components.
    ///
    /// Wraps `Path::canonicalize()` and converts the resulting `io::Error`
    /// into an `IncludeError::IoError`.
    fn canonicalize_path(&self, path: &Path) -> Result<PathBuf, IncludeError> {
        path.canonicalize().map_err(IncludeError::IoError)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    /// Helper: create a temporary directory structure for testing.
    /// Returns the root temp directory path.
    fn create_temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bcc_test_include_{}", prefix));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("failed to create temp dir");
        dir
    }

    /// Helper: create a file with the given content inside a directory.
    fn create_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("failed to create parent dirs");
        }
        let mut file = fs::File::create(&path).expect("failed to create file");
        file.write_all(content.as_bytes())
            .expect("failed to write file");
        path
    }

    /// Helper: clean up a temporary directory.
    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    // =======================================================================
    // IncludeKind tests
    // =======================================================================

    #[test]
    fn test_include_kind_equality() {
        assert_eq!(IncludeKind::Angle, IncludeKind::Angle);
        assert_eq!(IncludeKind::Quoted, IncludeKind::Quoted);
        assert_ne!(IncludeKind::Angle, IncludeKind::Quoted);
    }

    #[test]
    fn test_include_kind_clone_copy() {
        let kind = IncludeKind::Angle;
        let cloned = kind.clone();
        let copied = kind;
        assert_eq!(cloned, copied);
    }

    // =======================================================================
    // IncludeError tests
    // =======================================================================

    #[test]
    fn test_include_error_not_found_display() {
        let err = IncludeError::NotFound("missing.h".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("missing.h"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_include_error_circular_display() {
        let err = IncludeError::CircularInclude(PathBuf::from("/tmp/circular.h"));
        let msg = format!("{}", err);
        assert!(msg.contains("circular"));
    }

    #[test]
    fn test_include_error_pragma_once_display() {
        let err = IncludeError::PragmaOnce(PathBuf::from("/tmp/once.h"));
        let msg = format!("{}", err);
        assert!(msg.contains("pragma once"));
    }

    #[test]
    fn test_include_error_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let include_err: IncludeError = io_err.into();
        match include_err {
            IncludeError::IoError(e) => {
                assert_eq!(e.kind(), io::ErrorKind::NotFound);
            }
            _ => panic!("expected IoError variant"),
        }
    }

    // =======================================================================
    // IncludeResolver::new() tests
    // =======================================================================

    #[test]
    fn test_resolver_new_basic() {
        let resolver = IncludeResolver::new(
            vec![PathBuf::from("/usr/include")],
            Some(PathBuf::from("/opt/bcc/include")),
            vec![PathBuf::from("/usr/local/include")],
        );
        assert_eq!(resolver.include_dirs.len(), 1);
        assert!(resolver.bundled_header_path.is_some());
        assert_eq!(resolver.system_include_dirs.len(), 1);
        assert!(resolver.include_stack.is_empty());
        assert!(resolver.pragma_once_files.is_empty());
    }

    #[test]
    fn test_resolver_new_empty_dirs() {
        let resolver = IncludeResolver::new(vec![], None, vec![]);
        assert!(resolver.include_dirs.is_empty());
        assert!(resolver.system_include_dirs.is_empty());
        assert!(resolver.include_stack.is_empty());
    }

    // =======================================================================
    // resolve() tests — Quoted includes
    // =======================================================================

    #[test]
    fn test_resolve_quoted_from_current_dir() {
        let root = create_temp_dir("quoted_current");
        let current = root.join("project");
        fs::create_dir_all(&current).unwrap();
        create_file(&current, "local.h", "// local header");

        let resolver = IncludeResolver::new(vec![], None, vec![]);
        let result = resolver.resolve("local.h", IncludeKind::Quoted, &current);
        assert!(result.is_ok(), "should find local.h in current dir");
        let resolved = result.unwrap();
        assert!(resolved.ends_with("local.h"));

        cleanup(&root);
    }

    #[test]
    fn test_resolve_quoted_falls_through_to_i_dirs() {
        let root = create_temp_dir("quoted_fallthrough");
        let current = root.join("project");
        let inc_dir = root.join("extra_inc");
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&inc_dir).unwrap();
        create_file(&inc_dir, "extra.h", "// extra header");

        let resolver = IncludeResolver::new(vec![inc_dir], None, vec![]);
        // "extra.h" is not in current dir, should be found in -I dir
        let result = resolver.resolve("extra.h", IncludeKind::Quoted, &current);
        assert!(result.is_ok(), "should find extra.h in -I dir");

        cleanup(&root);
    }

    // =======================================================================
    // resolve() tests — Angle includes
    // =======================================================================

    #[test]
    fn test_resolve_angle_skips_current_dir() {
        let root = create_temp_dir("angle_skip");
        let current = root.join("project");
        fs::create_dir_all(&current).unwrap();
        create_file(&current, "only_here.h", "// only in current dir");

        let resolver = IncludeResolver::new(vec![], None, vec![]);
        let result = resolver.resolve("only_here.h", IncludeKind::Angle, &current);
        assert!(
            result.is_err(),
            "angle include should NOT search current dir"
        );
        match result.unwrap_err() {
            IncludeError::NotFound(name) => assert_eq!(name, "only_here.h"),
            _ => panic!("expected NotFound error"),
        }

        cleanup(&root);
    }

    #[test]
    fn test_resolve_angle_finds_in_i_dir() {
        let root = create_temp_dir("angle_i_dir");
        let current = root.join("project");
        let inc_dir = root.join("inc");
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&inc_dir).unwrap();
        create_file(&inc_dir, "system.h", "// system header");

        let resolver = IncludeResolver::new(vec![inc_dir], None, vec![]);
        let result = resolver.resolve("system.h", IncludeKind::Angle, &current);
        assert!(result.is_ok(), "angle include should find in -I dir");

        cleanup(&root);
    }

    // =======================================================================
    // resolve() tests — Search order and priority
    // =======================================================================

    #[test]
    fn test_resolve_i_dir_order_first_match_wins() {
        let root = create_temp_dir("i_dir_order");
        let current = root.join("project");
        let inc1 = root.join("inc1");
        let inc2 = root.join("inc2");
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&inc1).unwrap();
        fs::create_dir_all(&inc2).unwrap();
        create_file(&inc1, "shared.h", "// from inc1");
        create_file(&inc2, "shared.h", "// from inc2");

        let resolver = IncludeResolver::new(vec![inc1.clone(), inc2], None, vec![]);
        let result = resolver
            .resolve("shared.h", IncludeKind::Angle, &current)
            .unwrap();
        // The resolved path should be under inc1, not inc2
        let canonical_inc1 = inc1.canonicalize().unwrap();
        assert!(
            result.starts_with(&canonical_inc1),
            "first -I dir should win; resolved={:?}, expected prefix={:?}",
            result,
            canonical_inc1,
        );

        cleanup(&root);
    }

    #[test]
    fn test_resolve_bundled_headers() {
        let root = create_temp_dir("bundled");
        let current = root.join("project");
        let bundled = root.join("bundled_include");
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&bundled).unwrap();
        create_file(&bundled, "stddef.h", "// bundled stddef");
        create_file(&bundled, "stdint.h", "// bundled stdint");

        let resolver = IncludeResolver::new(vec![], Some(bundled), vec![]);
        let result = resolver.resolve("stddef.h", IncludeKind::Angle, &current);
        assert!(result.is_ok(), "should find bundled stddef.h");

        let result2 = resolver.resolve("stdint.h", IncludeKind::Quoted, &current);
        assert!(
            result2.is_ok(),
            "should find bundled stdint.h via quoted too"
        );

        cleanup(&root);
    }

    #[test]
    fn test_resolve_system_include_dirs() {
        let root = create_temp_dir("system_dirs");
        let current = root.join("project");
        let sys = root.join("sys_include");
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&sys).unwrap();
        create_file(&sys, "unistd.h", "// system unistd");

        let resolver = IncludeResolver::new(vec![], None, vec![sys]);
        let result = resolver.resolve("unistd.h", IncludeKind::Angle, &current);
        assert!(result.is_ok(), "should find in system include dir");

        cleanup(&root);
    }

    #[test]
    fn test_resolve_not_found() {
        let root = create_temp_dir("not_found");
        let current = root.join("project");
        fs::create_dir_all(&current).unwrap();

        let resolver = IncludeResolver::new(vec![], None, vec![]);
        let result = resolver.resolve("nonexistent.h", IncludeKind::Angle, &current);
        assert!(result.is_err());
        match result.unwrap_err() {
            IncludeError::NotFound(name) => assert_eq!(name, "nonexistent.h"),
            other => panic!("expected NotFound, got {:?}", other),
        }

        cleanup(&root);
    }

    // =======================================================================
    // push_include() / pop_include() / current_dir() tests
    // =======================================================================

    #[test]
    fn test_push_pop_include_stack() {
        let root = create_temp_dir("push_pop");
        let file_a = create_file(&root, "a.h", "// a");
        let file_b = create_file(&root, "b.h", "// b");

        let mut resolver = IncludeResolver::new(vec![], None, vec![]);

        // Push file A
        assert!(resolver.push_include(&file_a).is_ok());
        assert_eq!(resolver.include_stack.len(), 1);

        // Push file B
        assert!(resolver.push_include(&file_b).is_ok());
        assert_eq!(resolver.include_stack.len(), 2);

        // Pop file B
        resolver.pop_include();
        assert_eq!(resolver.include_stack.len(), 1);

        // Pop file A
        resolver.pop_include();
        assert!(resolver.include_stack.is_empty());

        cleanup(&root);
    }

    #[test]
    fn test_circular_include_detection() {
        let root = create_temp_dir("circular");
        let file_a = create_file(&root, "a.h", "// a");
        let file_b = create_file(&root, "b.h", "// b");

        let mut resolver = IncludeResolver::new(vec![], None, vec![]);

        // Push A, then B
        assert!(resolver.push_include(&file_a).is_ok());
        assert!(resolver.push_include(&file_b).is_ok());

        // Try to push A again — should detect circular include
        let result = resolver.push_include(&file_a);
        assert!(result.is_err());
        match result.unwrap_err() {
            IncludeError::CircularInclude(_) => { /* expected */ }
            other => panic!("expected CircularInclude, got {:?}", other),
        }

        cleanup(&root);
    }

    #[test]
    fn test_current_dir_returns_parent_of_top_of_stack() {
        let root = create_temp_dir("current_dir");
        let subdir = root.join("subdir");
        fs::create_dir_all(&subdir).unwrap();
        let file = create_file(&subdir, "header.h", "// header");

        let mut resolver = IncludeResolver::new(vec![], None, vec![]);
        assert!(resolver.push_include(&file).is_ok());

        let dir = resolver.current_dir();
        assert!(dir.is_some());
        // The current dir should be the canonical form of subdir
        let canonical_subdir = subdir.canonicalize().unwrap();
        assert_eq!(dir.unwrap().canonicalize().unwrap(), canonical_subdir,);

        resolver.pop_include();
        assert!(resolver.current_dir().is_none());

        cleanup(&root);
    }

    #[test]
    fn test_current_dir_empty_stack_returns_none() {
        let resolver = IncludeResolver::new(vec![], None, vec![]);
        assert!(resolver.current_dir().is_none());
    }

    // =======================================================================
    // Pragma once tests
    // =======================================================================

    #[test]
    fn test_mark_and_check_pragma_once() {
        let root = create_temp_dir("pragma_once");
        let file = create_file(&root, "once.h", "// once");

        let mut resolver = IncludeResolver::new(vec![], None, vec![]);

        // Initially not pragma-once'd
        assert!(!resolver.is_pragma_once(&file));

        // Mark it
        resolver.mark_pragma_once(&file);

        // Now it should be detected
        assert!(resolver.is_pragma_once(&file));

        cleanup(&root);
    }

    #[test]
    fn test_pragma_once_prevents_re_inclusion() {
        let root = create_temp_dir("pragma_prevent");
        let file = create_file(&root, "guarded.h", "// guarded");

        let mut resolver = IncludeResolver::new(vec![], None, vec![]);

        // First inclusion succeeds
        assert!(resolver.push_include(&file).is_ok());
        resolver.mark_pragma_once(&file);
        resolver.pop_include();

        // Second inclusion should return PragmaOnce error
        let result = resolver.push_include(&file);
        assert!(result.is_err());
        match result.unwrap_err() {
            IncludeError::PragmaOnce(_) => { /* expected */ }
            other => panic!("expected PragmaOnce, got {:?}", other),
        }

        cleanup(&root);
    }

    // =======================================================================
    // read_file() tests
    // =======================================================================

    #[test]
    fn test_read_file_success() {
        let root = create_temp_dir("read_file");
        let content = "int x = 42;\n";
        let file = create_file(&root, "source.c", content);

        let resolver = IncludeResolver::new(vec![], None, vec![]);
        let result = resolver.read_file(&file);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), content);

        cleanup(&root);
    }

    #[test]
    fn test_read_file_not_found() {
        let resolver = IncludeResolver::new(vec![], None, vec![]);
        let result = resolver.read_file(Path::new("/tmp/bcc_nonexistent_12345.h"));
        assert!(result.is_err());
        match result.unwrap_err() {
            IncludeError::IoError(e) => {
                assert_eq!(e.kind(), io::ErrorKind::NotFound);
            }
            other => panic!("expected IoError, got {:?}", other),
        }
    }

    #[test]
    fn test_read_file_empty_file() {
        let root = create_temp_dir("read_empty");
        let file = create_file(&root, "empty.h", "");

        let resolver = IncludeResolver::new(vec![], None, vec![]);
        let result = resolver.read_file(&file);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");

        cleanup(&root);
    }

    // =======================================================================
    // Multiple -I directories with correct priority
    // =======================================================================

    #[test]
    fn test_multiple_i_dirs_priority() {
        let root = create_temp_dir("multi_i");
        let current = root.join("project");
        let dir_a = root.join("dir_a");
        let dir_b = root.join("dir_b");
        let dir_c = root.join("dir_c");
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&dir_a).unwrap();
        fs::create_dir_all(&dir_b).unwrap();
        fs::create_dir_all(&dir_c).unwrap();

        // Only dir_b and dir_c have the file
        create_file(&dir_b, "test.h", "// from b");
        create_file(&dir_c, "test.h", "// from c");

        let resolver = IncludeResolver::new(vec![dir_a, dir_b.clone(), dir_c], None, vec![]);
        let result = resolver
            .resolve("test.h", IncludeKind::Angle, &current)
            .unwrap();
        let canonical_b = dir_b.canonicalize().unwrap();
        assert!(
            result.starts_with(&canonical_b),
            "dir_b should be found first; resolved={:?}",
            result,
        );

        cleanup(&root);
    }

    // =======================================================================
    // Quoted include falls through all search paths
    // =======================================================================

    #[test]
    fn test_quoted_include_full_search_chain() {
        let root = create_temp_dir("full_chain");
        let current = root.join("project");
        let inc = root.join("inc");
        let bundled = root.join("bundled");
        let sys = root.join("sys");
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&inc).unwrap();
        fs::create_dir_all(&bundled).unwrap();
        fs::create_dir_all(&sys).unwrap();

        // Place files at different levels
        create_file(&current, "local.h", "// local");
        create_file(&inc, "incdir.h", "// inc");
        create_file(&bundled, "stddef.h", "// bundled");
        create_file(&sys, "system.h", "// sys");

        let resolver = IncludeResolver::new(vec![inc], Some(bundled), vec![sys]);

        // Quoted: should find local.h in current dir
        assert!(resolver
            .resolve("local.h", IncludeKind::Quoted, &current)
            .is_ok());
        // Quoted: should find incdir.h in -I dir
        assert!(resolver
            .resolve("incdir.h", IncludeKind::Quoted, &current)
            .is_ok());
        // Quoted: should find stddef.h in bundled dir
        assert!(resolver
            .resolve("stddef.h", IncludeKind::Quoted, &current)
            .is_ok());
        // Quoted: should find system.h in sys dir
        assert!(resolver
            .resolve("system.h", IncludeKind::Quoted, &current)
            .is_ok());

        cleanup(&root);
    }

    // =======================================================================
    // Subdirectory includes (e.g., "subdir/header.h")
    // =======================================================================

    #[test]
    fn test_resolve_subdirectory_include() {
        let root = create_temp_dir("subdir_include");
        let current = root.join("project");
        let inc = root.join("inc");
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&inc.join("sys")).unwrap();
        create_file(&inc.join("sys"), "types.h", "// sys/types.h");

        let resolver = IncludeResolver::new(vec![inc], None, vec![]);
        let result = resolver.resolve("sys/types.h", IncludeKind::Angle, &current);
        assert!(result.is_ok(), "should resolve subdirectory include");

        cleanup(&root);
    }

    // =======================================================================
    // Edge case: empty name
    // =======================================================================

    #[test]
    fn test_resolve_empty_name() {
        let root = create_temp_dir("empty_name");
        let current = root.join("project");
        fs::create_dir_all(&current).unwrap();

        let resolver = IncludeResolver::new(vec![], None, vec![]);
        let result = resolver.resolve("", IncludeKind::Quoted, &current);
        // An empty name should result in NotFound (or possibly match a
        // directory, but our implementation checks for is_file())
        assert!(result.is_err());

        cleanup(&root);
    }

    // =======================================================================
    // Deeply nested include stack
    // =======================================================================

    #[test]
    fn test_deep_include_stack() {
        let root = create_temp_dir("deep_stack");
        let mut files = Vec::new();
        for i in 0..20 {
            let f = create_file(&root, &format!("deep_{}.h", i), "// deep");
            files.push(f);
        }

        let mut resolver = IncludeResolver::new(vec![], None, vec![]);
        for f in &files {
            assert!(resolver.push_include(f).is_ok());
        }
        assert_eq!(resolver.include_stack.len(), 20);

        // Pop all
        for _ in 0..20 {
            resolver.pop_include();
        }
        assert!(resolver.include_stack.is_empty());

        cleanup(&root);
    }

    // =======================================================================
    // Bundled header list verification
    // =======================================================================

    #[test]
    fn test_all_nine_bundled_headers_resolvable() {
        let root = create_temp_dir("nine_bundled");
        let current = root.join("project");
        let bundled = root.join("include");
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&bundled).unwrap();

        let bundled_headers = [
            "stddef.h",
            "stdint.h",
            "stdarg.h",
            "stdbool.h",
            "limits.h",
            "float.h",
            "stdalign.h",
            "stdnoreturn.h",
            "iso646.h",
        ];

        for header in &bundled_headers {
            create_file(&bundled, header, &format!("// bundled {}", header));
        }

        let resolver = IncludeResolver::new(vec![], Some(bundled), vec![]);

        for header in &bundled_headers {
            let result = resolver.resolve(header, IncludeKind::Angle, &current);
            assert!(
                result.is_ok(),
                "bundled header '{}' should be resolvable",
                header,
            );
        }

        cleanup(&root);
    }

    // =======================================================================
    // Pop on empty stack should not panic
    // =======================================================================

    #[test]
    fn test_pop_include_empty_stack_no_panic() {
        let mut resolver = IncludeResolver::new(vec![], None, vec![]);
        // Popping an empty stack should be a no-op, not a panic
        resolver.pop_include();
        assert!(resolver.include_stack.is_empty());
    }

    // =======================================================================
    // Resolve prefers current dir over -I for quoted includes
    // =======================================================================

    #[test]
    fn test_quoted_prefers_current_dir_over_i_dir() {
        let root = create_temp_dir("prefer_current");
        let current = root.join("current");
        let inc = root.join("inc");
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&inc).unwrap();

        // Same filename in both directories
        create_file(&current, "config.h", "// from current");
        create_file(&inc, "config.h", "// from inc");

        let resolver = IncludeResolver::new(vec![inc], None, vec![]);
        let result = resolver
            .resolve("config.h", IncludeKind::Quoted, &current)
            .unwrap();

        let canonical_current = current.canonicalize().unwrap();
        assert!(
            result.starts_with(&canonical_current),
            "quoted include should prefer current dir; resolved={:?}",
            result,
        );

        cleanup(&root);
    }
}
