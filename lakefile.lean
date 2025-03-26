import Lake
open Lake DSL

package «x1LeanDojo» {
  -- Hier können Build-Optionen definiert werden
}

lean_lib «Lean4Example» {
  -- Hier können Bibliotheksoptionen stehen
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

@[default_target]
lean_lib «Lean4Example»
