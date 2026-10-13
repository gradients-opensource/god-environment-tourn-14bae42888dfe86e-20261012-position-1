"""PVP_* memory/slot constants — copied from /ephemeral/G.O.D core/pvp/constants.py.

Only the constants relevant to SFT trajectory generation (memory slot sizing)
are copied; see §1.4 of docs/SFT_ALIGNMENT_PLAN.md. Keep values in sync with
core/pvp/constants.py — these size the memory block rendered by pvp_format.py
and must match what LLMBot.__init__ uses at eval time.
"""

PVP_WORKING_MEM_SLOTS = 4
PVP_WORKING_SLOT_TOKENS = 128
PVP_LONGTERM_MEM_SLOTS = 8
PVP_LONGTERM_SLOT_TOKENS = 128

# Generation caps — not used by training directly, copied for reference/parity.
PVP_TURN_MAX_TOKENS = 512
PVP_REFLECTION_MAX_TOKENS = 384
