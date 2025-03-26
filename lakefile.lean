import Lake
open Lake DSL

package «x1LeanDojo» {
  -- add package configuration options here
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

@[default_target]
lean_lib «Lean4Example» {
  -- add library configuration options here
}
