//! RISC-V 64-bit code generation backend.
//!
//! Implements the `CodeGen` trait for the LP64D ABI with RV64GC instruction
//! set encoding (32-bit base and optional 16-bit compressed).

pub mod abi;
pub mod encoding;
pub mod isel;

use crate::codegen::{Architecture, CodeGen, CodeGenError, ObjectCode};
use crate::driver::target::TargetConfig;
use crate::ir::Module;

/// RISC-V 64-bit code generation backend.
pub struct Riscv64CodeGen;

impl Riscv64CodeGen {
    /// Create a new RISC-V 64 code generator instance.
    pub fn new() -> Self {
        Self
    }
}

impl CodeGen for Riscv64CodeGen {
    fn generate(
        &self,
        _module: &Module,
        _target: &TargetConfig,
    ) -> Result<ObjectCode, CodeGenError> {
        // Stub: full implementation will be provided by the assigned agent.
        let object = ObjectCode::new(self.target_arch());
        Ok(object)
    }

    fn target_arch(&self) -> Architecture {
        Architecture::Riscv64
    }
}
