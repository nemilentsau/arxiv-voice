import unittest

from arxiv_voice.dialogue import DialogueTurn, parse_dialogue_script


class DialogueTests(unittest.TestCase):
    def test_parse_dialogue_script_extracts_turns(self) -> None:
        script = (
            "HOST: First point.\n\n"
            "COHOST: Question?\n"
            "More detail here.\n\n"
            "HOST: Answer."
        )
        turns = parse_dialogue_script(script)
        self.assertEqual(
            turns,
            [
                DialogueTurn("HOST", "First point."),
                DialogueTurn("COHOST", "Question? More detail here."),
                DialogueTurn("HOST", "Answer."),
            ],
        )

    def test_parse_dialogue_script_ignores_preamble_without_speaker(self) -> None:
        script = "Intro line\nHOST: Start here.\nCOHOST: Sure."
        turns = parse_dialogue_script(script)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].speaker, "HOST")


if __name__ == "__main__":
    unittest.main()
