import unittest


class UtilsTest(unittest.TestCase):
    def test_setup_logging(self):
        from spec_tools.utils.apps import setup_logging
        import logging
        setup_logging()
        self.assertEqual(logging.getLogger().level, logging.INFO)


if __name__ == '__main__':
    unittest.main()
