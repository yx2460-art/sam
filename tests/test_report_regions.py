import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from experiments.report_regions import save_png, select_cases


class VisualReportTests(unittest.TestCase):
    def test_selection_is_deterministic_and_does_not_invent_empty_examples(self):
        domain=np.ones((8,8),dtype=bool)
        harmful=np.zeros_like(domain)
        harmful[0,0]=harmful[4,4]=True
        cases=select_cases(harmful,np.zeros_like(domain),domain,4,2)
        self.assertEqual([(r['x'],r['y']) for r in cases],[(0,0),(4,4)])
        self.assertTrue(all(r['selected_for']=='harmful' for r in cases))
        self.assertEqual(select_cases(harmful,harmful,np.zeros_like(domain),4,2),[])

    def test_png_pixels_round_trip_through_standard_zlib_stream(self):
        data=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[1,2,3]]],dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'image.png'
            save_png(path,data)
            encoded=path.read_bytes()
            self.assertEqual(encoded[:8],b'\x89PNG\r\n\x1a\n')
            offset=8
            payload=b''
            while offset<len(encoded):
                length=struct.unpack('!I',encoded[offset:offset+4])[0]
                kind=encoded[offset+4:offset+8]
                chunk=encoded[offset+8:offset+8+length]
                crc=struct.unpack('!I',encoded[offset+8+length:offset+12+length])[0]
                self.assertEqual(crc,zlib.crc32(kind+chunk)&0xffffffff)
                if kind==b'IDAT':
                    payload+=chunk
                offset+=length+12
            decoded=zlib.decompress(payload)
            self.assertEqual(decoded,b'\0'+data[0].tobytes()+b'\0'+data[1].tobytes())
