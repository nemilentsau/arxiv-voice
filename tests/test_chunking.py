import unittest

from arxiv_voice.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_chunk_text_splits_large_input(self) -> None:
        text = "\n\n".join(
            [
                "Paragraph one " * 20,
                "Paragraph two " * 20,
                "Paragraph three " * 20,
            ]
        )
        chunks = chunk_text(text, max_chars=350)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_chunk_text_preserves_overlap(self) -> None:
        text = "\n\n".join(["alpha " * 40, "beta " * 40, "gamma " * 40])
        chunks = chunk_text(text, max_chars=250, overlap_chars=30)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[1].startswith(chunks[0][-30:].strip()))


if __name__ == "__main__":
    unittest.main()
