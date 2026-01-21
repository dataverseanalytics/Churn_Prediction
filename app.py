"""
FastAPI Backend for Membership Renewal Risk Assessment System
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from data_processor import DataProcessor
from risk_calculator import RiskCalculator
from ml_model import ChurnPredictor
import os

app = FastAPI(title="Membership Renewal Risk Assessment API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data processor
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
processor = DataProcessor(
    os.path.join(DATA_DIR, 'membership-records from Gomask.csv'),
    os.path.join(DATA_DIR, 'subscription-billing from GoMask.csv')
)

# Initialize risk calculator with default weights
calculator = RiskCalculator()

# Initialize ML churn predictor (will load trained model)
try:
    ml_predictor = ChurnPredictor()
    ml_available = True
    print("✓ ML Churn Predictor loaded successfully")
except Exception as e:
    ml_available = False
    ml_predictor = None
    print(f"⚠ ML Predictor not available: {str(e)}")
    print("  Run 'python model_trainer.py' to train the model first")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models for request/response
class RiskCalculationRequest(BaseModel):
    weights: Optional[Dict[str, float]] = None
    enabled_attributes: Optional[List[str]] = None
    member_ids: Optional[List[str]] = None


@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse('static/index.html')


@app.get("/api/members")
async def get_members():
    """
    Get all members who need risk assessment
    (unpaid invoices, individual bill-to only)
    """
    try:
        members_data = processor.prepare_for_risk_calculation()
        
        return {
            'success': True,
            'count': len(members_data),
            'members': members_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/attributes")
async def get_attributes():
    """Get list of available attributes and their default weights"""
    return {
        'success': True,
        'attributes': list(calculator.DEFAULT_WEIGHTS.keys()),
        'default_weights': calculator.DEFAULT_WEIGHTS
    }


@app.post("/api/calculate-risk")
async def calculate_risk(request: RiskCalculationRequest):
    """
    Calculate risk for members with custom weights and enabled attributes
    """
    try:
        # Get custom weights if provided
        weights = request.weights if request.weights else calculator.DEFAULT_WEIGHTS
        enabled_attributes = request.enabled_attributes
        member_ids = request.member_ids
        
        # Update calculator weights
        if weights:
            calculator.set_weights(weights)
        
        # Get members data
        members_data = processor.prepare_for_risk_calculation()
        
        # Filter by member_ids if provided
        if member_ids:
            members_data = [m for m in members_data if m['member_id'] in member_ids]
        
        # Calculate risk for all members
        results = calculator.calculate_batch_risk(members_data, enabled_attributes)
        
        # Calculate statistics
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        for result in results:
            risk_counts[result['risk_category']] += 1
        
        return {
            'success': True,
            'count': len(results),
            'results': results,
            'statistics': {
                'total': len(results),
                'low_risk': risk_counts['low'],
                'medium_risk': risk_counts['medium'],
                'high_risk': risk_counts['high']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/member/{member_id}")
async def get_member(member_id: str):
    """Get detailed information for a specific member"""
    try:
        members_data = processor.prepare_for_risk_calculation()
        
        member = next((m for m in members_data if m['member_id'] == member_id), None)
        
        if not member:
            raise HTTPException(status_code=404, detail='Member not found')
        
        # Calculate risk for this member
        result = calculator.calculate_member_risk(member)
        
        return {
            'success': True,
            'member': result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test")
async def test():
    """Test endpoint to verify API is working"""
    return {
        'success': True,
        'message': 'API is working!',
        'version': '2.0.0',
        'ml_available': ml_available
    }


@app.post("/api/ml/predict-churn")
async def predict_churn_ml(request: RiskCalculationRequest):
    """
    Predict churn using ML model
    """
    if not ml_available:
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Please train the model first by running: python model_trainer.py"
        )
    
    try:
        # Get members data
        members_data = processor.prepare_for_risk_calculation()
        
        # Filter by member_ids if provided
        if request.member_ids:
            members_data = [m for m in members_data if m['member_id'] in request.member_ids]
        
        # Prepare data for ML model (map to expected format)
        ml_predictions = []
        
        for member in members_data:
            # Map member data to ML model format
            ml_input = {
                'CustomerID': member['member_id'],
                'Age': member['attributes'].get('years_practicing', 0) + 25,  # Approximate age
                'Gender': 'Male',  # Default (not available in GoMask data)
                'Tenure': member['attributes'].get('membership_years', 0),
                'Usage Frequency': int(member['attributes'].get('website_activity', 0) / 3.33),  # Scale to 0-30
                'Support Calls': 0,  # Not available
                'Payment Delay': 0,  # Default to 0 as balance doesn't imply delay
                'Subscription Type': 'Premium' if member.get('membership_tier') == 'Premium' else 'Standard',
                'Contract Length': 'Annual',  # Default
                'Total Spend': member['attributes'].get('purchase_history', 0),
                'Last Interaction': int(100 - member['attributes'].get('website_activity', 50))  # Invert
            }
            # Get ML prediction
            prediction = ml_predictor.predict(ml_input)
            
            ml_predictions.append({
                'member_id': member['member_id'],
                'name': member['name'],
                'email': member['email'],
                'invoice_balance': member['invoice_balance'],
                'churn_probability': prediction['churn_probability'],
                'risk_category': prediction['risk_category'],
                'will_churn': prediction['will_churn']
            })
            
            # Debug: Print first feature vector to understand high risk scores
            if len(ml_predictions) > 0 and len(ml_predictions) <= 3:
                print(f"DEBUG ML Input for {member['member_id']}:")
                print(f"  Tenure: {ml_input['Tenure']}")
                print(f"  Usage: {ml_input['Usage Frequency']}")
                print(f"  Payment Delay: {ml_input['Payment Delay']}")
                print(f"  Prediction: {prediction['churn_probability']}%")

        # Calculate statistics
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        for pred in ml_predictions:
            risk_counts[pred['risk_category']] += 1
        
        return {
            'success': True,
            'model_type': 'machine_learning',
            'count': len(ml_predictions),
            'results': ml_predictions,
            'statistics': {
                'total': len(ml_predictions),
                'low_risk': risk_counts['low'],
                'medium_risk': risk_counts['medium'],
                'high_risk': risk_counts['high']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/feature-importance")
async def get_feature_importance():
    """
    Get feature importance from ML model
    """
    if not ml_available:
        raise HTTPException(
            status_code=503,
            detail="ML model not available"
        )
    
    try:
        importance = ml_predictor.get_feature_importance(top_n=15)
        return {
            'success': True,
            'feature_importance': importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/status")
async def ml_status():
    """Get ML model status"""
    return {
        'success': True,
        'ml_available': ml_available,
        'model_loaded': ml_predictor is not None,
        'message': 'ML model ready' if ml_available else 'ML model not trained. Run: python model_trainer.py'
    }


if __name__ == '__main__':
    import uvicorn
    
    print("=" * 60)
    print("Membership Renewal Risk Assessment System")
    print("=" * 60)
    print("\nStarting FastAPI server...")
    print("Access the application at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
