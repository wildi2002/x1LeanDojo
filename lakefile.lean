import Lake
open Lake DSL

require "leanprover-community" / "mathlib"

package «x1LeanDojo» {
  -- Hier können Build-Optionen definiert werden
}

@[default_target]
lean_lib «Lean4Example» {
  -- Hier können Bibliotheksoptionen stehen
}
