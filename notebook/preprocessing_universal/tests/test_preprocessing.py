"""Regression checks for semantic preservation, schema, adapters and optional modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'legacy'))
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from preprocessing import (Config, SLANG, approved_slang, audit_views, export_result,
                           model_view, near_duplicates, normalize_elongation,
                           normalize_slang, preprocess_frame, safe_clean, typo_candidates)


class PreprocessingTests(unittest.TestCase):
    def test_invalid_and_duplicate_index(self):
        raw = pd.DataFrame({'comment': [None, pd.NA, 123, ['a'], 'nan', 'gk'],
                            'label': range(6)}, index=[0, 0, 1, 1, 2, 2])
        before = raw.copy(deep=True)
        out = preprocess_frame(raw, 'comment')
        pd.testing.assert_frame_equal(raw, before)
        pd.testing.assert_series_equal(out.text_raw, raw.comment, check_names=False)
        pd.testing.assert_series_equal(out.label, raw.label)
        self.assertEqual(out.text_safe_cleaned.tolist(), ['', '', '', '', 'nan', 'gk'])
        self.assertEqual(out.is_non_string_text.tolist(), [False, False, True, True, False, False])

    def test_empty_frame(self):
        out = preprocess_frame(pd.DataFrame({'text': []}))
        self.assertEqual(len(out), 0)
        self.assertIn('text_social', out)
        self.assertEqual(len(audit_views(out)), 5)

    def test_unicode_emoji_and_negations(self):
        text = 'TIDAK BAGUS!!! 👩‍🍳 👨‍👩‍👧‍👦 🙏🏽 🇮🇩 massa Allah SPPG Bandung Barat Rp10.000'
        self.assertEqual(safe_clean(text), text)
        self.assertIn('tidak', model_view(text))
        self.assertEqual(safe_clean('cafÃ©'), 'café')
        self.assertEqual(safe_clean('Cafe\u0301'), 'Café')
        self.assertEqual(safe_clean('ａｂｃ'), 'ａｂｃ')

    def test_html_opt_in(self):
        self.assertEqual(safe_clean('nilai < 5 > 1'), 'nilai < 5 > 1')
        self.assertEqual(safe_clean('<p>A &amp; B</p><script>bad()</script>', Config(html_input=True)), 'A & B')
        self.assertEqual(safe_clean('a\r\nb', Config(keep_newlines=True)), 'a\nb')

    def test_lexical_boundaries(self):
        t = 'gk, bgs! https://x.id/gk @gk #gk a@gk.id SPPG sm dr'
        norm = normalize_slang(t, SLANG)
        self.assertEqual(norm, 'tidak, bagus! https://x.id/gk @gk #gk a@gk.id SPPG sm dr')
        self.assertEqual(normalize_elongation('baguuuuus bangeeet Allah massa Xxxx'), 'bagus banget Allah massa Xxxx')

    def test_external_conflict_and_protection(self):
        lex = pd.DataFrame({'variant': ['xyz', 'sm', 'sppg', 'zzz', 'zzz'],
                            'canonical': ['baik', 'sama', 'salah', 'satu', 'dua'],
                            'review_status': ['approved'] * 5})
        active = approved_slang(lex)
        self.assertEqual(active['xyz'], 'baik')
        for key in ['sm', 'sppg', 'zzz']:
            self.assertNotIn(key, active)

    def test_recipe_and_stopwords(self):
        self.assertEqual(model_view('@Budi BAGUS https://x.id', 'indobertweet'), '@USER bagus HTTPURL')
        self.assertEqual(model_view('dan tidak bukan jangan belum', 'topic'), 'tidak bukan jangan belum')
        with self.assertRaises(ValueError):
            model_view('a', 'typo-view')

    def test_duplicates_and_no_row_deletion(self):
        out = preprocess_frame(pd.DataFrame({'text': ['', None, 'gk', 'gk', 'tidak']}))
        self.assertEqual(len(out), 5)
        self.assertEqual(out.exact_duplicate_count.tolist(), [0, 0, 2, 2, 1])
        self.assertIsNone(out.duplicate_group_id.iloc[0])
        self.assertNotEqual(out.duplicate_group_id.iloc[2], out.duplicate_group_id.iloc[4])
        with self.assertRaises(ValueError):
            preprocess_frame(out)

    def test_typo_no_automatic_mutation(self):
        texts = ['maknan SPPG Bandung ora']
        suggestions = typo_candidates(texts, {'makanan': 100, 'bergizi': 50})
        self.assertIn('makanan', suggestions.canonical.tolist())
        self.assertEqual(texts, ['maknan SPPG Bandung ora'])

    def test_near_duplicate_verification(self):
        text = 'Program makan bergizi gratis untuk siswa sekolah sangat bermanfaat sekali'
        edges, report = near_duplicates([text, text + '!', text, 'teks berbeda total'], threshold=.8)
        self.assertEqual(report['unique_indexed'], 2)
        self.assertEqual(len(edges), 1)
        self.assertGreaterEqual(edges.jaccard.iloc[0], .8)

    def test_export_and_roundtrip(self):
        import json
        frame = preprocess_frame(pd.DataFrame({'text': ['gk @budi 😡', None], 'id': [1, 2]}))
        with tempfile.TemporaryDirectory() as tmp:
            folder = export_result(frame, tmp)
            loaded = pd.read_json(folder / 'comments.jsonl', lines=True)
            self.assertEqual(loaded.text_safe_cleaned.tolist(), frame.text_safe_cleaned.tolist())
            self.assertEqual(loaded.quality_flags.tolist(), frame.quality_flags.tolist())
            self.assertEqual(json.loads((folder / 'manifest.json').read_text())['rows'], 2)

    def test_stemming_optional(self):
        self.assertEqual(model_view('makanan tidak', cfg=Config(stem=True)), 'makan tidak')


if __name__ == '__main__':
    unittest.main(verbosity=2)
