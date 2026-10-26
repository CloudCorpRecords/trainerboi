"""Tests for data_quality: the shared dataset quality rules used by the
Dataset Builder and the Data Doctor.

data_quality is pure stdlib, so these tests keep the suite's
dependency-free contract (plain `python -m unittest`, no installs),
the same contract CI relies on.
"""
import unittest

import data_quality
from data_quality import check, grounded, parts, script_of


class PartsTest(unittest.TestCase):
    def test_messages_become_sft_with_last_turns(self):
        ex = {"messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ]}
        p = parts(ex)
        self.assertEqual(p["kind"], "sft")
        self.assertEqual(p["prompt"], "second question")
        self.assertEqual(p["response"], "second answer")

    def test_chosen_rejected_become_dpo(self):
        p = parts({"prompt": "q", "chosen": "the good answer", "rejected": "the bad answer"})
        self.assertEqual(p["kind"], "dpo")
        self.assertEqual(p["response"], "the good answer")
        self.assertEqual(p["rejected"], "the bad answer")

    def test_dpo_list_values_take_last_message(self):
        p = parts({"prompt": "q",
                   "chosen": [{"role": "assistant", "content": "final good"}],
                   "rejected": [{"role": "assistant", "content": "final bad"}]})
        self.assertEqual(p["response"], "final good")
        self.assertEqual(p["rejected"], "final bad")

    def test_prompt_completion_becomes_sft(self):
        p = parts({"prompt": "a prompt", "completion": "a completion"})
        self.assertEqual(p["kind"], "sft")
        self.assertEqual(p["prompt"], "a prompt")
        self.assertEqual(p["response"], "a completion")

    def test_text_key_becomes_text(self):
        p = parts({"text": "just some prose"})
        self.assertEqual(p["kind"], "text")
        self.assertEqual(p["text"], "just some prose")

    def test_non_dict_becomes_text(self):
        p = parts("plain string input")
        self.assertEqual(p["kind"], "text")
        self.assertEqual(p["text"], "plain string input")


class ScriptOfTest(unittest.TestCase):
    def test_latin(self):
        self.assertEqual(script_of("The quick brown fox jumps over the lazy dog"), "latin")

    def test_cyrillic(self):
        self.assertEqual(script_of("Привет, как дела у тебя сегодня"), "cyrillic")

    def test_greek(self):
        self.assertEqual(script_of("Καλημέρα, τι κάνεις σήμερα"), "greek")

    def test_cjk(self):
        self.assertEqual(script_of("今天天气很好我们去公园"), "cjk")

    def test_no_letters_is_none(self):
        self.assertIsNone(script_of("1234 !!! ???"))
        self.assertIsNone(script_of(""))


class CheckCleanTest(unittest.TestCase):
    def test_clean_sft_scores_100(self):
        score, flags = check({
            "prompt": "What is the capital of France?",
            "completion": "The capital of France is Paris, a city known for the Eiffel Tower.",
        })
        self.assertEqual(flags, [])
        self.assertEqual(score, 100)

    def test_clean_text_scores_100(self):
        score, flags = check({"text": "A short, ordinary paragraph about cooking rice at home for dinner tonight."})
        self.assertEqual(flags, [])
        self.assertEqual(score, 100)


class CheckFlagsTest(unittest.TestCase):
    def test_empty_text_scores_zero(self):
        score, flags = check({"text": "   "})
        self.assertEqual(flags, ["empty"])
        self.assertEqual(score, 0)

    def test_missing_response_is_empty(self):
        score, flags = check({"prompt": "A perfectly good question?", "completion": ""})
        self.assertIn("empty", flags)
        self.assertEqual(score, 0)

    def test_placeholder_answer_scores_zero(self):
        score, flags = check({"prompt": "What is two plus two?", "completion": "TODO"})
        self.assertEqual(flags, ["placeholder"])
        self.assertEqual(score, 0)

    def test_identical_pair_is_flagged(self):
        same = "The answer is forty two, worked out by adding the two numbers together."
        score, flags = check({"prompt": "Pick the better reply please", "chosen": same, "rejected": same})
        self.assertEqual(flags, ["pair_identical"])
        self.assertEqual(score, 0)

    def test_echo_answer(self):
        text = "Explain how photosynthesis works in green plants"
        score, flags = check({"prompt": text, "completion": text})
        self.assertEqual(flags, ["echo"])
        self.assertEqual(score, 40)

    def test_tiny_response(self):
        score, flags = check({"prompt": "Is the sky blue on a clear day?", "completion": "Yes."})
        self.assertEqual(flags, ["tiny_response"])
        self.assertEqual(score, 60)

    def test_short_prompt(self):
        score, flags = check({
            "prompt": "Hi",
            "completion": "Hello there, it is very nice to meet you today, how can I help?",
        })
        self.assertEqual(flags, ["short_prompt"])
        self.assertEqual(score, 80)

    def test_pii_in_prompt(self):
        score, flags = check({
            "prompt": "Please email jane.doe@example.com about the quarterly report",
            "completion": "I will send the quarterly report summary over to that address today.",
        })
        self.assertEqual(flags, ["pii"])
        self.assertEqual(score, 85)

    def test_ai_boilerplate(self):
        score, flags = check({
            "prompt": "Write a short poem about the sea",
            "completion": "As an AI language model, I cannot help with that request at all.",
        })
        self.assertEqual(flags, ["ai_boilerplate"])
        self.assertEqual(score, 65)

    def test_unclosed_code_fence_in_text(self):
        score, flags = check({"text": "Here is a longer piece of sample code for you:\n```python\nprint('hello there world')\n"})
        self.assertEqual(flags, ["code_fence"])
        self.assertEqual(score, 80)

    def test_repetitive_text(self):
        score, flags = check({"text": "the cat sat " * 20})
        self.assertEqual(flags, ["repetition"])
        self.assertEqual(score, 30)

    def test_mojibake_text(self):
        score, flags = check({"text": "The cafÃ© near the station serves lunch every day of the week."})
        self.assertEqual(flags, ["mojibake"])
        self.assertEqual(score, 40)

    def test_language_mismatch(self):
        score, flags = check({
            "prompt": "Please describe your home town in some detail for me",
            "completion": "Мой родной город находится на берегу большой реки, там много парков, старых улиц и красивых мостов через воду.",
        })
        self.assertEqual(flags, ["language_mismatch"])
        self.assertEqual(score, 60)

    def test_truncated_answer(self):
        resp = ("The history of the railway began in the early nineteenth century, when steam "
                "locomotives first carried passengers between industrial towns across the "
                "north of England and then spread quickly over the")
        self.assertGreaterEqual(len(resp), 120)
        score, flags = check({"prompt": "Tell me about the history of railways", "completion": resp})
        self.assertEqual(flags, ["truncated"])
        self.assertEqual(score, 70)

    def test_scores_stay_within_bounds(self):
        samples = [
            {"text": ""},
            {"text": "TODO"},
            {"prompt": "q", "completion": "a"},
            {"prompt": "What is two plus two?", "completion": "TODO"},
            {"text": "the cat sat " * 20},
        ]
        for ex in samples:
            score, flags = check(ex)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
            self.assertEqual(score, max(0, 100 - sum(data_quality.FLAGS[f][0] for f in flags)))


class FlagsTableTest(unittest.TestCase):
    def test_flags_are_well_formed(self):
        for name, entry in data_quality.FLAGS.items():
            self.assertEqual(len(entry), 3, name)
            penalty, label, drop = entry
            self.assertGreaterEqual(penalty, 1, name)
            self.assertLessEqual(penalty, 100, name)
            self.assertTrue(label, name)
            self.assertIsInstance(drop, bool, name)

    def test_blockers_drop_by_default(self):
        for name in ("empty", "placeholder", "pair_identical"):
            penalty, _label, drop = data_quality.FLAGS[name]
            self.assertEqual(penalty, 100, name)
            self.assertTrue(drop, name)


class GroundedTest(unittest.TestCase):
    CHUNK = "The sky appears blue on a clear day because sunlight is scattered by air molecules."
    QUESTION = "Why does the sky look blue on a clear day?"

    def test_supported_answer_is_grounded(self):
        self.assertTrue(grounded(self.QUESTION, "The sky appears blue because sunlight is scattered by air molecules.", self.CHUNK))

    def test_unrelated_answer_is_not_grounded(self):
        self.assertFalse(grounded(self.QUESTION, "To cook pasta, boil salted water and stir the noodles often.", self.CHUNK))

    def test_invented_number_is_not_grounded(self):
        chunk = "The village sits in a valley and is home to a few hundred families and a small school."
        self.assertFalse(grounded("How many people live in the village?", "The village has exactly 42000 residents living in it.", chunk))

    def test_answer_without_content_words_is_not_grounded(self):
        self.assertFalse(grounded(self.QUESTION, "It is so.", self.CHUNK))


if __name__ == "__main__":
    unittest.main()
