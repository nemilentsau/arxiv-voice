import unittest

from arxiv_voice.dialogue import (
    DialogueTurn,
    chunk_dialogue_turns,
    coalesce_dialogue_turns,
    parse_dialogue_script,
    render_dia_script,
)


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

    def test_coalesce_dialogue_turns_merges_adjacent_turns(self) -> None:
        turns = [
            DialogueTurn("HOST", "One."),
            DialogueTurn("HOST", "Two."),
            DialogueTurn("COHOST", "Three."),
        ]
        merged = coalesce_dialogue_turns(turns)
        self.assertEqual(
            merged,
            [
                DialogueTurn("HOST", "One. Two."),
                DialogueTurn("COHOST", "Three."),
            ],
        )

    def test_chunk_dialogue_turns_splits_batches(self) -> None:
        turns = [DialogueTurn("HOST" if i % 2 == 0 else "COHOST", str(i)) for i in range(12)]
        chunks = chunk_dialogue_turns(turns, chunk_size=10)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 10)
        self.assertEqual(len(chunks[1]), 2)

    def test_render_dia_script_converts_speakers(self) -> None:
        text = render_dia_script(
            [
                DialogueTurn("HOST", "Hello there."),
                DialogueTurn("COHOST", "Hi."),
            ]
        )
        self.assertEqual(text, "[S1] Hello there.\n[S2] Hi.")


if __name__ == "__main__":
    unittest.main()
