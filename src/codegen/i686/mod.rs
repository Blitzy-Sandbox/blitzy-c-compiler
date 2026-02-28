//! i686 (32-bit x86) code generation backend.
//!
//! Implements the `CodeGen` trait for the cdecl ABI with 32-bit instruction
//! encoding (no REX prefix) and register pair arithmetic for 64-bit operations.

pub mod abi;
pub mod encoding;
pub mod isel;

use crate::codegen::{Architecture, CodeGen, CodeGenError, ObjectCode};
use crate::driver::target::TargetConfig;
use crate::ir::Module;

/// i686 code generation backend.
pub struct I686CodeGen;

impl I686CodeGen {
    /// Create a new i686 code generator instance.
    pub fn new() -> Self {
        Self
    }
}

impl CodeGen for I686CodeGen {
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
        Architecture::I686
    }
}
