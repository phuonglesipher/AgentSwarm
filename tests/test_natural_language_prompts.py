from __future__ import annotations

import unittest

from core.natural_language_prompts import build_llm_request, build_prompt_brief


class NaturalLanguagePromptTests(unittest.TestCase):
    def test_build_prompt_brief_skips_null_like_sections_system_wide(self) -> None:
        prompt = build_prompt_brief(
            opening="Plan the next safe gameplay step.",
            sections=[
                ("Optional context", None),
                ("Empty bullets", "- None."),
                ("Existing owner", "src/combat_runtime.py"),
                ("Retry count", 0),
                ("", "Keep the charge cap unchanged."),
            ],
            closing="None.",
        )

        self.assertIn("Plan the next safe gameplay step.", prompt)
        self.assertIn("## Existing owner\nsrc/combat_runtime.py", prompt)
        self.assertIn("## Retry count\n0", prompt)
        self.assertIn("Keep the charge cap unchanged.", prompt)
        self.assertNotIn("## Optional context", prompt)
        self.assertNotIn("## Empty bullets", prompt)
        self.assertNotIn("None.", prompt)

    def test_build_llm_request_omits_null_like_context_block(self) -> None:
        prompt = build_llm_request(
            instructions="Review the grounded gameplay change.",
            input_text="- None.",
            require_structured_output=True,
        )

        self.assertIn("Review the grounded gameplay change.", prompt)
        self.assertNotIn("Here is the current working context:", prompt)
        self.assertIn("Respond through the configured structured output channel", prompt)


if __name__ == "__main__":
    unittest.main()
