from lean_dojo import LeanGame

# Beispiel für einen Theorem-Beweis in Lean
game = LeanGame("theorem_name")
state_before = game.get_state()

# Dein Modell schlägt eine Taktik vor
proposed_tactic = "rw [← Nat.mod_add_div (2 * m + 1) 8]"

# Überprüfung in Lean
valid = game.apply_tactic(proposed_tactic)

if valid:
    reward = 1  # Positiv bewerten
else:
    reward = -1  # Negativ bewerten
