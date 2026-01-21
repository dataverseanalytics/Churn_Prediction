"""
Risk Calculator Module
Implements the probability-based risk scoring algorithm
"""

from typing import Dict, List
import numpy as np


class RiskCalculator:
    """Calculate membership renewal risk based on weighted attributes"""
    
    # Default weights for each attribute (equal distribution)
    DEFAULT_WEIGHTS = {
        'committee_participation': 12.5,
        'membership_years': 12.5,
        'meeting_attendance': 12.5,
        'purchase_history': 12.5,
        'donation_activity': 12.5,
        'years_practicing': 12.5,
        'previously_lapsed': 12.5,
        'website_activity': 12.5
    }
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize risk calculator
        
        Args:
            weights: Dictionary of attribute weights (must sum to 100)
        """
        self.weights = weights if weights else self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()
    
    def _validate_weights(self):
        """Ensure weights sum to 100%"""
        total = sum(self.weights.values())
        if not (99.9 <= total <= 100.1):  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 100%, got {total}%")
    
    def set_weights(self, weights: Dict[str, float]):
        """Update weights for calculation"""
        self.weights = weights
        self._validate_weights()
    
    def normalize_attribute(self, value: float, attr_name: str, 
                           min_val: float = 0, max_val: float = 100) -> float:
        """
        Normalize attribute value to 0-100 scale
        
        Different attributes have different scales, so we normalize them
        """
        if attr_name == 'committee_participation':
            # Engagement score is already 0-100
            return min(100, max(0, value))
        
        elif attr_name == 'membership_years':
            # 0-20 years normalized to 0-100
            return min(100, (value / 20) * 100)
        
        elif attr_name == 'meeting_attendance':
            # Engagement score is already 0-100
            return min(100, max(0, value))
        
        elif attr_name == 'purchase_history':
            # Normalize based on typical range (0-5000)
            return min(100, (value / 5000) * 100)
        
        elif attr_name == 'donation_activity':
            # Binary: 0 or 100
            return value * 100
        
        elif attr_name == 'years_practicing':
            # 0-40 years normalized to 0-100
            return min(100, (value / 40) * 100)
        
        elif attr_name == 'previously_lapsed':
            # Binary but INVERTED (lapsed = bad = 0, not lapsed = good = 100)
            return 0 if value == 1.0 else 100
        
        elif attr_name == 'website_activity':
            # Already 0-100 from recency calculation
            return min(100, max(0, value))
        
        return value
    
    def calculate_risk_score(self, attributes: Dict[str, float], 
                            enabled_attributes: List[str] = None) -> float:
        """
        Calculate risk score based on weighted attributes
        Requirement 4: Perform probability calculation with weighted percentages
        
        Args:
            attributes: Dictionary of attribute values
            enabled_attributes: List of attributes to include (None = all)
        
        Returns:
            Risk score as percentage (0-100)
        """
        if enabled_attributes is None:
            enabled_attributes = list(self.weights.keys())
        
        # Adjust weights for enabled attributes only
        active_weights = {k: v for k, v in self.weights.items() 
                         if k in enabled_attributes}
        
        if not active_weights:
            return 0.0
        
        # Normalize weights to sum to 100
        total_weight = sum(active_weights.values())
        normalized_weights = {k: (v / total_weight) * 100 
                             for k, v in active_weights.items()}
        
        # Calculate weighted score
        total_score = 0.0
        for attr_name, weight in normalized_weights.items():
            if attr_name in attributes:
                # Normalize the attribute value
                normalized_value = self.normalize_attribute(
                    attributes[attr_name], 
                    attr_name
                )
                
                # Add weighted contribution
                contribution = (normalized_value * weight) / 100
                total_score += contribution
        
        # INVERT the score: High engagement = Low risk
        # So we return (100 - score) as the risk percentage
        risk_score = 100 - total_score
        
        return round(risk_score, 2)
    
    def categorize_risk(self, risk_score: float) -> str:
        """
        Categorize risk level based on score
        Requirement 5: Categorize as low, medium, or high risk
        
        Args:
            risk_score: Risk percentage (0-100)
        
        Returns:
            Risk category: 'low', 'medium', or 'high'
        """
        if risk_score <= 30:
            return 'low'
        elif risk_score <= 60:
            return 'medium'
        else:
            return 'high'
    
    def calculate_member_risk(self, member_data: Dict, 
                             enabled_attributes: List[str] = None) -> Dict:
        """
        Calculate complete risk assessment for a member
        
        Args:
            member_data: Dictionary with member info and attributes
            enabled_attributes: List of attributes to include
        
        Returns:
            Dictionary with risk score, category, and details
        """
        attributes = member_data.get('attributes', {})
        
        risk_score = self.calculate_risk_score(attributes, enabled_attributes)
        risk_category = self.categorize_risk(risk_score)
        
        return {
            'member_id': member_data.get('member_id'),
            'name': member_data.get('name'),
            'email': member_data.get('email'),
            'risk_score': risk_score,
            'risk_category': risk_category,
            'invoice_balance': member_data.get('invoice_balance'),
            'attributes': attributes
        }
    
    def calculate_batch_risk(self, members_data: List[Dict], 
                            enabled_attributes: List[str] = None) -> List[Dict]:
        """
        Calculate risk for multiple members
        
        Args:
            members_data: List of member dictionaries
            enabled_attributes: List of attributes to include
        
        Returns:
            List of risk assessment results
        """
        results = []
        for member in members_data:
            result = self.calculate_member_risk(member, enabled_attributes)
            results.append(result)
        
        # Sort by risk score (highest first)
        results.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return results


if __name__ == "__main__":
    # Test the risk calculator
    calculator = RiskCalculator()
    
    # Sample member data
    test_member = {
        'member_id': 'M0001',
        'name': 'Test Member',
        'email': 'test@gmail.com',
        'invoice_balance': 50.0,
        'attributes': {
            'committee_participation': 75.0,
            'membership_years': 5.0,
            'meeting_attendance': 80.0,
            'purchase_history': 500.0,
            'donation_activity': 1.0,
            'years_practicing': 10.0,
            'previously_lapsed': 0.0,
            'website_activity': 85.0
        }
    }
    
    result = calculator.calculate_member_risk(test_member)
    print("\nRisk Assessment Result:")
    print(f"Member: {result['name']}")
    print(f"Risk Score: {result['risk_score']}%")
    print(f"Risk Category: {result['risk_category'].upper()}")
    print(f"Invoice Balance: ${result['invoice_balance']}")
