//! AArch64 (ARM 64-bit) code generation backend.
//!
//! Implements the `CodeGen` trait for the AAPCS64 ABI with fixed-width
//! 32-bit instruction encoding.

pub mod abi;
pub mod encoding;
pub mod isel;

use crate::codegen::{Architecture, CodeGen, CodeGenError, ObjectCode};
use crate::driver::target::TargetConfig;
use crate::ir::Module;

/// AArch64 code generation backend.
pub struct Aarch64CodeGen;

impl Aarch64CodeGen {
    /// Create a new AArch64 code generator instance.
    pub fn new() -> Self {
        Self
    }
}

impl CodeGen for Aarch64CodeGen {
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
        Architecture::Aarch64
    }
}
