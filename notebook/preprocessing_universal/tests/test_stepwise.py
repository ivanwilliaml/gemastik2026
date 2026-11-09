"""Checks for step-by-step text pipeline, independent of notebook UI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from stepwise import *


class StepwiseTests(unittest.TestCase):
    def test_input_raw_types_and_order(self):
        raw = pd.Series([None, pd.NA, 123, ['x'], '', 'nan', 'None'], index=[4, 4, 1, 8, 9, 2, 0])
        out = preprocess_texts(raw)
        pd.testing.assert_series_equal(out.text_raw, raw, check_names=False)
        self.assertEqual(out.text_safe.tolist(), ['', '', '', '', '', 'nan', 'None'])
        self.assertEqual(out.index.tolist(), raw.index.tolist())

    def test_empty(self):
        out = preprocess_texts(pd.Series([], dtype='string'))
        self.assertEqual(len(out), 0)
        self.assertIn('tokens', out)

    def test_safe_semantics(self):
        text = 'TIDAK BAGUS!!! 👩‍🍳 👨‍👩‍👧‍👦 🙏🏽 🇮🇩 Allah massa SPPG Bandung Rp10.000'
        out = preprocess_texts([text])
        self.assertEqual(out.text_safe[0], text)
        self.assertIn('tidak', out.text_clean[0].split())
        self.assertIn('sppg', out.text_clean[0].split())
        self.assertIn('😡', preprocess_texts(['gk 😡']).text_normalized[0])

    def test_stage_order_repeatability(self):
        raw = pd.Series(['gk bagus!!! 😡', 'baguuuuus anak2', None])
        a, b = preprocess_texts(raw), preprocess_texts(raw)
        pd.testing.assert_frame_equal(a, b)
        texts = raw
        for branch, steps in build_steps().items():
            for _, function in steps:
                texts = transform_unique(texts, function)
            pd.testing.assert_series_equal(a['text_' + branch], texts, check_names=False)

    def test_lexical_boundaries_and_protection(self):
        t = 'gk, https://x.id/gk a@gk.id @gk #gk sm dr SPPG'
        self.assertEqual(normalize_slang(t), 'tidak, https://x.id/gk a@gk.id @gk #gk sm dr SPPG')
        self.assertEqual(normalize_elongation('baguuuuus Allah massa xyztttt'), 'bagus Allah massa xyztttt')
        self.assertEqual(apply_reviewed_typos('SPPG maknan', {'sppg': 'salah', 'maknan': 'makanan'}), 'SPPG makanan')

    def test_encoding_html_controls(self):
        self.assertEqual(repair_encoding('cafÃ©'), 'café')
        self.assertEqual(normalize_unicode('Cafe\u0301'), 'Café')
        self.assertEqual(normalize_unicode('ａｂｃ'), 'ａｂｃ')
        self.assertEqual(remove_invisible('teks\u200b 👩‍🍳'), 'teks 👩‍🍳')
        self.assertEqual(remove_html('<p>A &amp; B</p><script>bad()</script>'), 'A & B')
        self.assertEqual(preprocess_texts(['nilai < 5 > 1']).text_safe[0], 'nilai < 5 > 1')
        self.assertEqual(decode_byte_escapes(r'\xf0\x9f\x98\xa1'), '😡')
        self.assertEqual(decode_byte_escapes(r'\xff'), r'\xff')

    def test_social_elements(self):
        self.assertEqual(replace_urls('cek https://x.id/x, ya'), 'cek  URL , ya')
        self.assertIn('EMAIL', replace_emails('a@contoh.id'))
        self.assertIn('PHONE', replace_phones('+62 812-3456-7890'))
        self.assertEqual(replace_phones('3 siswa Rp10.000 10/11/2026'), '3 siswa Rp10.000 10/11/2026')
        self.assertEqual(segment_hashtags('#DukungProgramMBG'), ' Dukung Program MBG ')
        self.assertEqual(normalize_reduplication('anak2 S2 GPT2'), 'anak-anak S2 GPT2')

    def test_options_and_views(self):
        text = 'ora apik 3 siswa maknan 😡'
        out = preprocess_texts([text], {'regional': True, 'remove_numbers': True, 'stem': False,
                                       'typos': {'maknan': 'makanan'}})
        self.assertIn('tidak bagus', out.text_normalized[0])
        self.assertIn('makanan', out.text_clean[0])
        self.assertNotIn('3', out.text_clean[0])
        self.assertIn('3', out.text_normalized[0])
        self.assertIn('😡', out.text_normalized[0])
        self.assertNotIn('😡', out.text_clean[0])

    def test_negation_stopword_and_stemming(self):
        self.assertEqual(remove_stopwords('dan tidak bukan belum jangan sppg', {'dan', 'tidak', 'bukan', 'belum', 'jangan', 'sppg'}),
                         'tidak bukan belum jangan sppg')
        self.assertEqual(stem_text('makanan tidak sppg'), 'makan tidak sppg')
        self.assertEqual(indobertweet_view('@Budi BAGUS https://x.id'), '@USER bagus HTTPURL')

    def test_no_concatenation(self):
        self.assertEqual(remove_punctuation('tidak/bagus'), 'tidak bagus')
        self.assertEqual(normalize_obfuscation('k3r4cUn4n S2 Rp10.000'), 'keracunan S2 Rp10.000')
        self.assertEqual(reduce_repeated_words('bagus bagus banget'), 'bagus banget')


if __name__ == '__main__':
    unittest.main(verbosity=2)
