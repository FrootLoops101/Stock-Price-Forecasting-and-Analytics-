import unittest
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import relevant classes/functions from your project
from main import EnhancedStockForecaster, generate_enhanced_stock_data

class TestStockSense(unittest.TestCase):
    def setUp(self):
        self.symbol = 'TSLA'
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime(2024, 1, 1)
        self.forecaster = EnhancedStockForecaster()
        # Generate and load data
        stock_data = generate_enhanced_stock_data(self.symbol, self.start_date, self.end_date)
        self.forecaster.stock_data[self.symbol] = stock_data

    def test_stock_data_generation(self):
        data = self.forecaster.stock_data[self.symbol]
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 1000)
        self.assertIn('Close', data.columns)
        self.assertTrue((data['Close'] > 0).all())

    def test_calculate_technical_indicators(self):
        data = self.forecaster.calculate_technical_indicators(self.symbol)
        # Check key indicators
        for col in ['SMA_20', 'EMA_12', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower']:
            self.assertIn(col, data.columns)
            self.assertFalse(data[col].isnull().all())
        
    def test_calculate_risk_metrics(self):
        self.forecaster.calculate_technical_indicators(self.symbol)
        metrics = self.forecaster.calculate_risk_metrics(self.symbol)
        self.assertIsInstance(metrics, dict)
        self.assertIn('Sharpe Ratio', metrics)
        self.assertIn('Max Drawdown', metrics)
        self.assertTrue(metrics['Max Drawdown'] <= 0)
    
    def test_ml_training_and_prediction(self):
        self.forecaster.calculate_technical_indicators(self.symbol)
        model_results, ensemble_pred = self.forecaster.train_ml_models(self.symbol)
        self.assertIsInstance(model_results, dict)
        self.assertGreater(len(model_results), 0)
        # Check Random Forest results presence
        self.assertIn('Random Forest', model_results)
        rf_result = model_results['Random Forest']
        self.assertTrue('mae' in rf_result)
        self.assertGreater(rf_result['mae'], 0)
        self.assertEqual(len(ensemble_pred), len(self.forecaster.ml_results[self.symbol]['y_test']))

    def test_visualization_functions(self):
        self.forecaster.calculate_technical_indicators(self.symbol)
        # Smoke test to ensure plotting functions run without error
        try:
            self.forecaster.create_comprehensive_chart(self.symbol)
            self.forecaster.create_ml_visualization(self.symbol)
            self.forecaster.create_portfolio_dashboard()
        except Exception as e:
            self.fail(f'Visualization function raised an exception: {e}')
        
if __name__ == '__main__':
    unittest.main()
