import unittest

from database.text_processor import TextProcessor


class TextProcessorDoiExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an existing path to avoid creating new directories during tests.
        cls.processor = TextProcessor(data_dir="data/raw")

    def test_extracts_plain_doi_label(self):
        content = "# Title\n\nDOI: 10.1002/CCTC.202200897\n"
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")

    def test_extracts_doi_url(self):
        content = "Supporting information: https://doi.org/10.1002/cctc.202200897\n"
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")

    def test_extracts_dx_doi_url(self):
        content = "Link: http://dx.doi.org/10.1002/cctc.202200897\n"
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")

    def test_extracts_markdown_link_and_strips_punctuation(self):
        content = "See [paper](https://doi.org/10.1002/cctc.202200897)."
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")

    def test_extracts_angle_bracket_url(self):
        content = "DOI: <https://doi.org/10.1002/cctc.202200897>"
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")

    def test_prefers_header_doi_over_reference_like_doi(self):
        content = (
            "# Title\n\n"
            "DOI: 10.1002/cctc.202200897\n\n"
            "## References\n"
            "- Ref A https://doi.org/10.1021/acs.jacs.0c00000\n"
        )
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")


if __name__ == "__main__":
    unittest.main()

