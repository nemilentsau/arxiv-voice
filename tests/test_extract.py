import unittest

from arxiv_voice.extract import extract_sections, normalize_text


class ExtractTests(unittest.TestCase):
    def test_normalize_text_collapses_extra_newlines(self) -> None:
        self.assertEqual(normalize_text("A\r\n\r\n\r\nB"), "A\n\nB")

    def test_extract_sections_finds_common_headings(self) -> None:
        text = "Abstract\n\n1 Introduction\n\n2 Method\n\nReferences"
        sections = extract_sections(text)
        headings = [item["heading"] for item in sections]
        self.assertIn("Abstract", headings)
        self.assertIn("1 Introduction", headings)
        self.assertIn("References", headings)


if __name__ == "__main__":
    unittest.main()
