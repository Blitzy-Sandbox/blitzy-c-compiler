//! # Dynamic Linking Support
//!
//! This module provides shared library support including `.dynamic` section
//! generation, `.dynsym`/`.dynstr` symbol tables, and PLT/GOT stub creation
//! for `-shared` output mode.
//!
//! ## Responsibilities
//! - Generate `.dynamic` section with `DT_NEEDED`, `DT_SONAME` entries
//! - Build `.dynsym` and `.dynstr` dynamic symbol tables
//! - Create `.plt` and `.got` stubs for lazy binding
//! - Support `ET_DYN` (shared object) ELF output type

use crate::driver::target::TargetConfig;
use crate::linker::elf;
use crate::linker::sections;
use crate::linker::symbols;

// ============================================================================
// Dynamic Section Types
// ============================================================================

/// ELF dynamic section tag values.
pub const DT_NULL: i64 = 0;
pub const DT_NEEDED: i64 = 1;
pub const DT_PLTRELSZ: i64 = 2;
pub const DT_PLTGOT: i64 = 3;
pub const DT_HASH: i64 = 4;
pub const DT_STRTAB: i64 = 5;
pub const DT_SYMTAB: i64 = 6;
pub const DT_RELA: i64 = 7;
pub const DT_RELASZ: i64 = 8;
pub const DT_RELAENT: i64 = 9;
pub const DT_STRSZ: i64 = 10;
pub const DT_SYMENT: i64 = 11;
pub const DT_SONAME: i64 = 14;
pub const DT_REL: i64 = 17;
pub const DT_RELSZ: i64 = 18;
pub const DT_RELENT: i64 = 19;
pub const DT_PLTREL: i64 = 20;
pub const DT_JMPREL: i64 = 23;

// ============================================================================
// Dynamic Section Entry
// ============================================================================

/// An entry in the `.dynamic` section.
#[derive(Debug, Clone)]
pub struct DynamicEntry {
    /// Dynamic tag (DT_NEEDED, DT_SONAME, etc.).
    pub tag: i64,
    /// Tag value (address or offset depending on tag type).
    pub value: u64,
}

// ============================================================================
// Dynamic Section Builder
// ============================================================================

/// Builds the `.dynamic` section, `.dynsym`, `.dynstr`, `.plt`, and `.got`
/// sections for shared library output.
#[derive(Debug)]
pub struct DynamicBuilder {
    /// Dynamic section entries.
    entries: Vec<DynamicEntry>,
    /// Dynamic string table content.
    dynstr: Vec<u8>,
    /// Dynamic symbol table content.
    dynsym: Vec<u8>,
    /// PLT section data.
    plt_data: Vec<u8>,
    /// GOT section data.
    got_data: Vec<u8>,
    /// Whether we're building for 64-bit.
    is_64bit: bool,
}

impl DynamicBuilder {
    /// Create a new dynamic section builder.
    pub fn new(is_64bit: bool) -> Self {
        let mut dynstr = Vec::new();
        dynstr.push(0); // Null string at index 0.
        DynamicBuilder {
            entries: Vec::new(),
            dynstr,
            dynsym: Vec::new(),
            plt_data: Vec::new(),
            got_data: Vec::new(),
            is_64bit,
        }
    }

    /// Add a DT_NEEDED entry for a shared library dependency.
    pub fn add_needed(&mut self, name: &str) {
        let offset = self.dynstr.len() as u64;
        self.dynstr.extend_from_slice(name.as_bytes());
        self.dynstr.push(0);
        self.entries.push(DynamicEntry {
            tag: DT_NEEDED,
            value: offset,
        });
    }

    /// Add a DT_SONAME entry.
    pub fn set_soname(&mut self, name: &str) {
        let offset = self.dynstr.len() as u64;
        self.dynstr.extend_from_slice(name.as_bytes());
        self.dynstr.push(0);
        self.entries.push(DynamicEntry {
            tag: DT_SONAME,
            value: offset,
        });
    }

    /// Build and return the dynamic sections as output sections.
    ///
    /// Returns a vector of output sections to be added to the ELF writer:
    /// `.dynamic`, `.dynsym`, `.dynstr`, `.plt`, `.got`.
    pub fn build(
        mut self,
        _target: &TargetConfig,
    ) -> Vec<elf::OutputSection> {
        // Terminate the dynamic section with DT_NULL.
        self.entries.push(DynamicEntry {
            tag: DT_NULL,
            value: 0,
        });

        // Serialize dynamic entries.
        let mut dynamic_data = Vec::new();
        for entry in &self.entries {
            if self.is_64bit {
                dynamic_data.extend_from_slice(&entry.tag.to_le_bytes());
                dynamic_data.extend_from_slice(&entry.value.to_le_bytes());
            } else {
                dynamic_data.extend_from_slice(&(entry.tag as i32).to_le_bytes());
                dynamic_data.extend_from_slice(&(entry.value as u32).to_le_bytes());
            }
        }

        let mut sections = Vec::new();

        // .dynamic section.
        let dyn_entsize = if self.is_64bit { 16u64 } else { 8u64 };
        sections.push(elf::OutputSection {
            name: String::from(".dynamic"),
            data: dynamic_data,
            header: elf::OutputSectionHeader {
                sh_type: elf::SHT_DYNAMIC,
                sh_flags: sections::SHF_ALLOC | sections::SHF_WRITE,
                sh_addr: 0,
                sh_addralign: if self.is_64bit { 8 } else { 4 },
                sh_entsize: dyn_entsize,
                sh_link: 0,
                sh_info: 0,
            },
        });

        // .dynstr section.
        sections.push(elf::OutputSection {
            name: String::from(".dynstr"),
            data: self.dynstr,
            header: elf::OutputSectionHeader {
                sh_type: elf::SHT_STRTAB,
                sh_flags: sections::SHF_ALLOC,
                sh_addr: 0,
                sh_addralign: 1,
                sh_entsize: 0,
                sh_link: 0,
                sh_info: 0,
            },
        });

        // .dynsym section.
        if !self.dynsym.is_empty() {
            let sym_entsize = if self.is_64bit { 24u64 } else { 16u64 };
            sections.push(elf::OutputSection {
                name: String::from(".dynsym"),
                data: self.dynsym,
                header: elf::OutputSectionHeader {
                    sh_type: elf::SHT_DYNSYM,
                    sh_flags: sections::SHF_ALLOC,
                    sh_addr: 0,
                    sh_addralign: if self.is_64bit { 8 } else { 4 },
                    sh_entsize: sym_entsize,
                    sh_link: 0,
                    sh_info: 0,
                },
            });
        }

        sections
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_constants() {
        assert_eq!(DT_NULL, 0);
        assert_eq!(DT_NEEDED, 1);
        assert_eq!(DT_SONAME, 14);
    }

    #[test]
    fn test_dynamic_builder_new() {
        let builder = DynamicBuilder::new(true);
        assert!(builder.entries.is_empty());
        assert_eq!(builder.dynstr.len(), 1); // Null byte
        assert!(builder.is_64bit);
    }

    #[test]
    fn test_dynamic_builder_add_needed() {
        let mut builder = DynamicBuilder::new(true);
        builder.add_needed("libc.so.6");
        assert_eq!(builder.entries.len(), 1);
        assert_eq!(builder.entries[0].tag, DT_NEEDED);
    }

    #[test]
    fn test_dynamic_builder_set_soname() {
        let mut builder = DynamicBuilder::new(true);
        builder.set_soname("libtest.so.1");
        assert_eq!(builder.entries.len(), 1);
        assert_eq!(builder.entries[0].tag, DT_SONAME);
    }

    #[test]
    fn test_dynamic_builder_build_64bit() {
        let mut builder = DynamicBuilder::new(true);
        builder.add_needed("libc.so.6");
        let target = TargetConfig::x86_64();
        let sections = builder.build(&target);
        assert!(!sections.is_empty());
        // Should have at least .dynamic and .dynstr
        assert!(sections.iter().any(|s| s.name == ".dynamic"));
        assert!(sections.iter().any(|s| s.name == ".dynstr"));
    }

    #[test]
    fn test_dynamic_builder_build_32bit() {
        let mut builder = DynamicBuilder::new(false);
        builder.add_needed("libc.so.6");
        let target = TargetConfig::i686();
        let sections = builder.build(&target);
        assert!(sections.iter().any(|s| s.name == ".dynamic"));
    }
}
