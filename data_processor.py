"""
Data Processor Module
Handles loading, merging, and preprocessing of membership and billing data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple


class DataProcessor:
    """Process membership and billing data for risk assessment"""
    
    def __init__(self, membership_file: str, billing_file: str):
        """
        Initialize the data processor
        
        Args:
            membership_file: Path to membership records CSV
            billing_file: Path to subscription billing CSV
        """
        self.membership_file = membership_file
        self.billing_file = billing_file
        self.members_df = None
        self.billing_df = None
        self.merged_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load CSV files into DataFrames"""
        try:
            self.members_df = pd.read_csv(self.membership_file)
            self.billing_df = pd.read_csv(self.billing_file)
            print(f"Loaded {len(self.members_df)} member records")
            print(f"Loaded {len(self.billing_df)} billing records")
            return self.members_df, self.billing_df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def merge_data(self) -> pd.DataFrame:
        """Merge membership and billing data"""
        if self.members_df is None or self.billing_df is None:
            self.load_data()
        
        # Create a mapping from member_id to customer email for joining
        # Since billing uses customer_email and members use email
        self.merged_df = pd.merge(
            self.members_df,
            self.billing_df,
            left_on='email',
            right_on='customer_email',
            how='left'
        )
        
        print(f"Merged data: {len(self.merged_df)} records")
        return self.merged_df
    
    def calculate_invoice_balance(self, row) -> float:
        """
        Calculate invoice balance based on payment status
        Requirement 1.1: balance = 0 if paid, > 0 if unpaid
        
        NOTE: If no billing data exists (payment_status is NaN), 
        assume unpaid with balance of 100 to show all members.
        """
        payment_status = row.get('payment_status', 'unknown')
        
        # If no billing data (NaN), assume unpaid
        if pd.isna(payment_status) or payment_status == 'unknown':
            return 100.0
        
        if payment_status == 'success':
            return 0.0
        elif payment_status in ['failed', 'pending']:
            # Return the plan price as unpaid balance
            return float(row.get('plan_price', 100))
        else:
            return 100.0
    
    def is_individual_bill_to(self, email: str) -> bool:
        """
        Determine if BILL TO is an individual or company
        Requirement 2: Check if email is individual (personal) or company
        
        Personal email domains: gmail, yahoo, outlook, hotmail, icloud, live, etc.
        Company emails: have company domain
        
        NOTE: Currently returns True for ALL emails to show all members.
        """
        if pd.isna(email):
            return False
        
        # DISABLED FILTER: Return True for all valid emails
        # This allows club/organization emails to be processed
        return True
    
    def calculate_age(self, date_of_birth: str) -> int:
        """Calculate age from date of birth"""
        if pd.isna(date_of_birth):
            return 0
        
        try:
            dob = pd.to_datetime(date_of_birth)
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age
        except:
            return 0
    
    def calculate_membership_years(self, start_date: str) -> float:
        """Calculate years of membership"""
        if pd.isna(start_date):
            return 0.0
        
        try:
            start = pd.to_datetime(start_date)
            today = datetime.now()
            years = (today - start).days / 365.25
            return round(years, 2)
        except:
            return 0.0
    
    def extract_member_attributes(self, row) -> Dict:
        """
        Extract member attributes for risk calculation
        Requirement 3: Gather member attributes
        """
        attributes = {
            # Committee history - using engagement_score as proxy
            'committee_participation': float(row.get('engagement_score', 0)),
            
            # Membership history - years of membership
            'membership_years': self.calculate_membership_years(row.get('membership_start_date')),
            
            # Meeting attendance - engagement score and recency
            'meeting_attendance': float(row.get('engagement_score', 0)),
            
            # Purchase history - total payments and amount
            'purchase_history': float(row.get('total_amount_paid', 0)),
            
            # Donation history - check engagement notes for donations
            'donation_activity': 1.0 if 'donat' in str(row.get('engagement_notes', '')).lower() else 0.0,
            
            # Years of practicing - calculate from age (assuming practice since age 25)
            'years_practicing': max(0, self.calculate_age(row.get('date_of_birth')) - 25),
            
            # Previously lapsed - check membership status
            'previously_lapsed': 1.0 if row.get('membership_status') == 'Lapsed' else 0.0,
            
            # Website login history - days since last engagement
            'website_activity': self._calculate_login_recency(row.get('last_engagement_date'))
        }
        
        return attributes
    
    def _calculate_login_recency(self, last_engagement_date: str) -> float:
        """Calculate recency score based on last engagement (higher = more recent)"""
        if pd.isna(last_engagement_date):
            return 0.0
        
        try:
            last_engagement = pd.to_datetime(last_engagement_date)
            today = datetime.now()
            days_ago = (today - last_engagement).days
            
            # Convert to score: recent = high score, old = low score
            # 0 days = 100, 365+ days = 0
            score = max(0, 100 - (days_ago / 3.65))
            return round(score, 2)
        except:
            return 0.0
    
    def filter_for_risk_assessment(self) -> pd.DataFrame:
        """
        Filter members who need risk assessment
        Requirements 1 & 2: Only unpaid invoices for individuals
        """
        if self.merged_df is None:
            self.merge_data()
        
        # Calculate invoice balance for each row
        self.merged_df['invoice_balance'] = self.merged_df.apply(
            self.calculate_invoice_balance, axis=1
        )
        
        # Determine if individual
        self.merged_df['is_individual'] = self.merged_df['email'].apply(
            self.is_individual_bill_to
        )
        
        # Filter: balance > 0 AND is individual
        filtered_df = self.merged_df[
            (self.merged_df['invoice_balance'] > 0) & 
            (self.merged_df['is_individual'] == True)
        ].copy()
        
        print(f"Filtered to {len(filtered_df)} members needing risk assessment")
        return filtered_df
    
    def prepare_for_risk_calculation(self) -> List[Dict]:
        """
        Prepare member data with attributes for risk calculation
        Returns list of member dictionaries with all necessary data
        """
        filtered_df = self.filter_for_risk_assessment()
        
        members_data = []
        for _, row in filtered_df.iterrows():
            member_data = {
                'member_id': row.get('member_id'),
                'name': f"{row.get('first_name', '')} {row.get('last_name', '')}",
                'email': row.get('email'),
                'membership_tier': row.get('membership_tier'),
                'invoice_balance': row.get('invoice_balance'),
                'attributes': self.extract_member_attributes(row)
            }
            members_data.append(member_data)
        
        return members_data


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor(
        'membership-records from Gomask.csv',
        'subscription-billing from GoMask.csv'
    )
    
    members = processor.prepare_for_risk_calculation()
    print(f"\nPrepared {len(members)} members for risk assessment")
    
    if members:
        print("\nSample member data:")
        print(f"ID: {members[0]['member_id']}")
        print(f"Name: {members[0]['name']}")
        print(f"Email: {members[0]['email']}")
        print(f"Balance: ${members[0]['invoice_balance']}")
        print(f"Attributes: {members[0]['attributes']}")
