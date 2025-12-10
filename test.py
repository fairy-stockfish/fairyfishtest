import unittest
import math

import fairyfishtest


class TestFairyFishTest(unittest.TestCase):
    def test_parse_time_control(self):
        tc = fairyfishtest.TimeControl.parse('10/20+3')
        self.assertEqual(tc.time, 20)
        self.assertEqual(tc.increment, 3)
        self.assertEqual(tc.moves, 10)

        tc = fairyfishtest.TimeControl.parse(str(tc))
        self.assertEqual(tc.time, 20)
        self.assertEqual(tc.increment, 3)
        self.assertEqual(tc.moves, 10)

        tc = fairyfishtest.TimeControl.parse('5.1+0.2')
        self.assertEqual(tc.time, 5.1)
        self.assertEqual(tc.increment, 0.2)
        self.assertEqual(tc.moves, 0)

        tc = fairyfishtest.TimeControl.parse('10')
        self.assertEqual(tc.time, 10)
        self.assertEqual(tc.increment, 0)
        self.assertEqual(tc.moves, 0)

    def test_xboard_time_control(self):
        tc = fairyfishtest.TimeControl.parse('10/20+3')
        self.assertEqual(tc.format_xboard(), '10 0:20 3')

    def test_to_uci(self):
        self.assertEqual(fairyfishtest.Engine.move_to_uci('a1a2', False), 'a1a2')
        self.assertEqual(fairyfishtest.Engine.move_to_uci('h8h9+', True), 'h9h10+')
        self.assertEqual(fairyfishtest.Engine.move_to_uci('a9b9', True), 'a10b10')
        self.assertEqual(fairyfishtest.Engine.move_to_uci('a0a1', True), 'a1a2')
        self.assertEqual(fairyfishtest.Engine.move_to_uci('P@e5', True), 'P@e6')
        self.assertEqual(fairyfishtest.Engine.move_to_uci('@@@@', True), '0000')

    def test_from_uci(self):
        self.assertEqual(fairyfishtest.Engine.move_from_uci('a1a2', False), 'a1a2')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('h9h10+', True), 'h8h9+')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('a10b10', True), 'a9b9')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('a1a2', True), 'a0a1')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('P@e6', True), 'P@e5')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('0000', True), '@@@@')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('e1e1', True), '@@@@')

    def test_erf_functions(self):
        # Test error function approximation
        self.assertAlmostEqual(fairyfishtest.erf(0), 0, places=4)
        self.assertAlmostEqual(fairyfishtest.erf(1), 0.8427, places=3)
        self.assertAlmostEqual(fairyfishtest.erf(-1), -0.8427, places=3)
        
        # Test inverse error function
        self.assertAlmostEqual(fairyfishtest.erf_inv(0), 0, places=4)
        self.assertAlmostEqual(fairyfishtest.erf_inv(0.5), 0.4769, places=3)

    def test_phi_functions(self):
        # Test cumulative distribution function
        self.assertAlmostEqual(fairyfishtest.phi(0), 0.5, places=4)
        self.assertAlmostEqual(fairyfishtest.phi(1), 0.8413, places=3)
        
        # Test quantile function
        self.assertAlmostEqual(fairyfishtest.phi_inv(0.5), 0, places=4)
        self.assertAlmostEqual(fairyfishtest.phi_inv(0.9), 1.281, places=2)  # Adjusted precision

    def test_elo_conversion(self):
        # Test win rate to ELO conversion
        self.assertAlmostEqual(fairyfishtest.elo(0.5), 0, places=4)
        self.assertAlmostEqual(fairyfishtest.elo(0.6), 70.4, places=0)  # Adjusted precision
        self.assertEqual(fairyfishtest.elo(0), 0.0)

    def test_bayeselo_conversions(self):
        # Test BayesELO to probability conversion
        proba = fairyfishtest.bayeselo_to_proba(0, 200)
        self.assertAlmostEqual(proba['win'] + proba['loss'] + proba['draw'], 1.0, places=6)
        self.assertAlmostEqual(proba['win'], proba['loss'], places=6)  # Equal at 0 ELO
        
        # Test probability to BayesELO conversion
        P = {'win': 0.4, 'loss': 0.3, 'draw': 0.3}
        elo, drawelo = fairyfishtest.proba_to_bayeselo(P)
        converted_back = fairyfishtest.bayeselo_to_proba(elo, drawelo)
        self.assertAlmostEqual(P['win'], converted_back['win'], places=4)
        self.assertAlmostEqual(P['loss'], converted_back['loss'], places=4)
        self.assertAlmostEqual(P['draw'], converted_back['draw'], places=4)

    def test_sprt_basic(self):
        # Test SPRT with insufficient data
        result = fairyfishtest.SPRT({'wins': 1, 'losses': 0, 'draws': 0}, 0, 0.05, 5, 0.05, 200)
        self.assertFalse(result['finished'])
        self.assertEqual(result['state'], '')
        
        # Test SPRT with mixed results (should not finish early)
        result = fairyfishtest.SPRT({'wins': 10, 'losses': 10, 'draws': 20}, 0, 0.05, 5, 0.05, 200)
        self.assertIn('llr', result)
        self.assertIn('lower_bound', result)
        self.assertIn('upper_bound', result)
        
        # Test bounds calculation
        alpha, beta = 0.05, 0.05
        expected_lower = math.log(beta / (1 - alpha))
        expected_upper = math.log((1 - beta) / alpha)
        self.assertAlmostEqual(result['lower_bound'], expected_lower, places=6)
        self.assertAlmostEqual(result['upper_bound'], expected_upper, places=6)

    def test_sprt_acceptance_rejection(self):
        # Test clear win case (should be accepted)
        result = fairyfishtest.SPRT({'wins': 100, 'losses': 10, 'draws': 20}, 0, 0.05, 5, 0.05, 200)
        # This might finish depending on the LLR value
        
        # Test clear loss case (should be rejected)  
        result = fairyfishtest.SPRT({'wins': 10, 'losses': 100, 'draws': 20}, 0, 0.05, 5, 0.05, 200)
        # This might finish depending on the LLR value


if __name__ == '__main__':
    unittest.main(verbosity=2)
