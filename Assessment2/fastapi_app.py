from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import pickle
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Segmentation API",
    description="Predict customer clusters and generate personalized marketing offers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



kmeans_model = None
scaler = None
label_encoders = None
feature_names = None
cluster_mapping = None

try:
    with open('customer_cluster_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    logger.info("✓ KMeans model loaded successfully")
except FileNotFoundError:
    logger.error("Error: customer_cluster_model.pkl not found")

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("✓ Scaler loaded successfully")
except FileNotFoundError:
    logger.error("Error: scaler.pkl not found")

try:
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    logger.info("✓ Label encoders loaded successfully")
except FileNotFoundError:
    logger.error("Error: label_encoders.pkl not found")

try:
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    logger.info("✓ Feature names loaded successfully")
except FileNotFoundError:
    logger.error("Error: feature_names.pkl not found")

try:
    with open('cluster_mapping.pkl', 'rb') as f:
        cluster_mapping = pickle.load(f)
    logger.info("✓ Cluster mapping loaded successfully")
except FileNotFoundError:
    logger.error("Error: cluster_mapping.pkl not found")



CLUSTER_DESCRIPTIONS = {
    0: {
        "name": "High-Value Loyal Customers",
        "description": "Premium customers with exceptional spending power and engagement",
        "characteristics": [
            "Highest average spending",
            "Strong app engagement",
            "High-value purchases",
            "Loyal & consistent buyers"
        ]
    },
    1: {
        "name": "Value-Seeking Regular Customers",
        "description": "Regular customers with moderate spending who seek good value",
        "characteristics": [
            "Consistent purchase frequency",
            "Moderate spending levels",
            "Balanced engagement",
            "Price-value conscious"
        ]
    },
    2: {
        "name": "Price-Sensitive Occasional Customers",
        "description": "Occasional buyers with lower engagement and spending",
        "characteristics": [
            "Low app usage",
            "Minimal spending",
            "Deal-driven purchases",
            "Sporadic engagement"
        ]
    }
}

PERSONALIZED_OFFERS = {
    0: {
        "discount": "15% Premium Loyalty Discount",
        "offers": [
            "Exclusive VIP access to new products",
            "Priority customer support (24/7)",
            "Free shipping on all orders",
            "Double loyalty points on purchases",
            "Exclusive member-only sales events"
        ],
        "strategy": "Maintain premium tier status with exclusive VIP benefits and white-glove service",
        "priority": "HIGH"
    },
    1: {
        "discount": "10% Loyalty Discount",
        "offers": [
            "Seasonal promotions and discounts",
            "Buy more save more offers",
            "Referral bonus program (₹500 per referral)",
            "Birthday special discounts",
            "Newsletter exclusive deals"
        ],
        "strategy": "Encourage regular purchases through value-based promotions and loyalty rewards",
        "priority": "MEDIUM"
    },
    2: {
        "discount": "20% First Purchase / Bundle Offers",
        "offers": [
            "Flash sales and limited-time offers",
            "Bundle deals (3 items at special price)",
            "Seasonal clearance sales",
            "Free shipping on minimum purchase ₹500+",
            "Try-before-you-buy incentives"
        ],
        "strategy": "Focus on aggressive discounts, volume deals, and low-commitment offerings to increase engagement",
        "priority": "LOW"
    }
}



class CustomerInput(BaseModel):
    """Customer input data for clustering prediction"""
    age: float = Field(..., description="Customer age (18-100)", ge=18, le=100)
    gender: str = Field(..., description="Gender: 'M' or 'F'", pattern="^[MF]$")
    annual_income: float = Field(..., description="Annual income in ₹", ge=10000)
    total_spent: float = Field(..., description="Total amount spent in ₹", ge=0)
    avg_order_value: float = Field(..., description="Average order value in ₹", ge=10)
    monthly_purchases: float = Field(..., description="Number of purchases per month", ge=0)
    discount_usage: str = Field(..., description="Discount usage level: 'Low', 'Medium', or 'High'", 
                                 pattern="^(Low|Medium|High)$")
    app_time_minutes: float = Field(..., description="App usage time per month in minutes", ge=0)
    preferred_shopping_time: str = Field(..., description="Preferred shopping time: 'Day' or 'Night'", 
                                          pattern="^(Day|Night)$")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "gender": "M",
                "annual_income": 80000,
                "total_spent": 15000,
                "avg_order_value": 1500,
                "monthly_purchases": 8,
                "discount_usage": "Medium",
                "app_time_minutes": 120,
                "preferred_shopping_time": "Night"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for cluster prediction"""
    customer_id: str = Field(..., description="Generated customer ID")
    cluster: int = Field(..., description="Assigned cluster (0, 1, or 2)")
    cluster_name: str = Field(..., description="Human-readable cluster name")
    confidence_score: float = Field(..., description="Prediction confidence (0-100%)")
    segment_description: str = Field(..., description="Detailed segment description")
    personalized_offers: Dict = Field(..., description="Customized offers for this customer")
    business_insights: Dict = Field(..., description="Actionable business insights")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    customers: List[CustomerInput] = Field(..., description="List of customers to predict")



def encode_customer_input(customer: CustomerInput) -> np.ndarray:
    """
    Encode categorical variables and prepare input for model prediction
    
    Args:
        customer: CustomerInput object
        
    Returns:
        Scaled numpy array ready for model prediction
    """
    if label_encoders is None:
        raise ValueError("Label encoders not loaded")
    
    # Encode categorical variables
    gender_encoded = label_encoders['le_gender'].transform([customer.gender])[0]
    discount_encoded = label_encoders['le_discount'].transform([customer.discount_usage])[0]
    time_encoded = label_encoders['le_time'].transform([customer.preferred_shopping_time])[0]
    
    # Create feature vector in correct order
    input_data = np.array([[
        customer.age,
        gender_encoded,
        customer.annual_income,
        customer.total_spent,
        customer.avg_order_value,
        customer.monthly_purchases,
        discount_encoded,
        customer.app_time_minutes,
        time_encoded
    ]])
    
    return input_data

def calculate_confidence(distances: np.ndarray, cluster_idx: int) -> float:
    """Calculate confidence score based on distance to cluster center"""
    min_distance = distances.min()
    max_distance = distances.max()
    
    if max_distance == 0:
        return 100.0
    
   
    confidence = ((max_distance - distances[cluster_idx]) / max_distance) * 100
    return round(confidence, 2)



@app.get("/", tags=["Root"])
def root():
    """Root endpoint - API information"""
    return {
        "message": "Customer Segmentation Clustering API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "alternative_docs": "/redoc"
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check endpoint"""
    models_status = {
        "kmeans_model": kmeans_model is not None,
        "scaler": scaler is not None,
        "label_encoders": label_encoders is not None,
        "feature_names": feature_names is not None,
        "cluster_mapping": cluster_mapping is not None
    }
    
    all_loaded = all(models_status.values())
    
    return {
        "status": "healthy" if all_loaded else "degraded",
        "models_loaded": models_status,
        "api_version": "1.0.0",
        "endpoint": "All models ready for predictions" if all_loaded else "Some models missing"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_customer_cluster(customer: CustomerInput):
    """
    Predict customer cluster and generate personalized marketing insights.
    
    This endpoint:
    - Encodes categorical variables
    - Scales the input features
    - Predicts the cluster using K-Means
    - Calculates confidence score
    - Returns personalized offers and business insights
    
    Args:
        customer: Customer information including demographics and behavior
        
    Returns:
        PredictionResponse with cluster prediction, offers, and insights
    """
    
    # Verify models are loaded
    if kmeans_model is None or scaler is None or label_encoders is None:
        raise HTTPException(
            status_code=503,
            detail="Models not fully loaded. Check model files: customer_cluster_model.pkl, scaler.pkl, label_encoders.pkl"
        )
    
    try:
        # Encode input
        input_data = encode_customer_input(customer)
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Predict cluster
        cluster = int(kmeans_model.predict(input_scaled)[0])
        
        # Calculate confidence
        distances = np.linalg.norm(input_scaled - kmeans_model.cluster_centers_, axis=1)
        confidence = calculate_confidence(distances, cluster)
        
        # Get cluster information
        cluster_name = CLUSTER_DESCRIPTIONS[cluster]['name']
        cluster_desc = CLUSTER_DESCRIPTIONS[cluster]['description']
        offers = PERSONALIZED_OFFERS[cluster]
        
        # Business insights
        insights = {
            "priority_level": offers['priority'],
            "recommended_strategy": offers['strategy'],
            "ltv_category": "Premium" if cluster == 0 else ("Standard" if cluster == 1 else "Budget"),
            "engagement_level": "High" if cluster == 0 else ("Medium" if cluster == 1 else "Low"),
            "distance_to_cluster_center": round(float(distances[cluster]), 4)
        }
        
        return PredictionResponse(
            customer_id=f"CUST_{np.random.randint(100000, 999999)}",
            cluster=cluster,
            cluster_name=cluster_name,
            confidence_score=confidence,
            segment_description=cluster_desc,
            personalized_offers={
                "discount": offers['discount'],
                "top_offers": offers['offers'],
                "marketing_strategy": offers['strategy']
            },
            business_insights=insights
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing prediction: {str(e)}"
        )

@app.post("/predict/batch", tags=["Prediction"])
def batch_predict(request: BatchPredictionRequest):
    """
    Perform batch predictions for multiple customers.
    
    Args:
        request: BatchPredictionRequest containing list of customers
        
    Returns:
        Dictionary with batch processing results
    """
    
    if kmeans_model is None or scaler is None or label_encoders is None:
        raise HTTPException(
            status_code=503,
            detail="Models not fully loaded"
        )
    
    results = []
    successful = 0
    failed = 0
    
    for idx, customer in enumerate(request.customers):
        try:
            # Encode and predict
            input_data = encode_customer_input(customer)
            input_scaled = scaler.transform(input_data)
            cluster = int(kmeans_model.predict(input_scaled)[0])
            
            # Calculate confidence
            distances = np.linalg.norm(input_scaled - kmeans_model.cluster_centers_, axis=1)
            confidence = calculate_confidence(distances, cluster)
            
            # Get offers
            offers = PERSONALIZED_OFFERS[cluster]
            
            results.append({
                "index": idx,
                "status": "success",
                "cluster": cluster,
                "cluster_name": CLUSTER_DESCRIPTIONS[cluster]['name'],
                "confidence": confidence,
                "discount_offer": offers['discount'],
                "top_3_offers": offers['offers'][:3]
            })
            successful += 1
        
        except Exception as e:
            logger.error(f"Batch prediction error at index {idx}: {str(e)}")
            results.append({
                "index": idx,
                "status": "failed",
                "error": str(e)
            })
            failed += 1
    
    return {
        "batch_id": f"BATCH_{np.random.randint(100000, 999999)}",
        "total_submitted": len(request.customers),
        "successful_predictions": successful,
        "failed_predictions": failed,
        "success_rate": round((successful / len(request.customers)) * 100, 2) if request.customers else 0,
        "results": results
    }

@app.get("/clusters/info", tags=["Information"])
def get_cluster_information():
    """
    Get detailed information about all customer clusters.
    
    Returns:
        Dictionary with cluster profiles, descriptions, and offers
    """
    
    cluster_info = {}
    
    for cluster_id in [0, 1, 2]:
        desc = CLUSTER_DESCRIPTIONS[cluster_id]
        offers = PERSONALIZED_OFFERS[cluster_id]
        
        cluster_info[f"Cluster_{cluster_id}"] = {
            "name": desc['name'],
            "description": desc['description'],
            "characteristics": desc['characteristics'],
            "discount": offers['discount'],
            "sample_offers": offers['offers'][:3],
            "marketing_strategy": offers['strategy'],
            "priority": offers['priority']
        }
    
    return cluster_info

@app.get("/features", tags=["Information"])
def get_feature_information():
    """
    Get information about model features.
    
    Returns:
        List of features and their descriptions
    """
    
    feature_descriptions = {
        "age": "Customer age (years)",
        "gender": "Customer gender (M/F)",
        "annual_income": "Annual income in ₹",
        "total_spent": "Total amount spent in ₹",
        "avg_order_value": "Average order value in ₹",
        "monthly_purchases": "Number of purchases per month",
        "discount_usage": "Discount usage level (Low/Medium/High)",
        "app_time_minutes": "App usage time per month in minutes",
        "preferred_shopping_time": "Preferred shopping time (Day/Night)"
    }
    
    return {
        "total_features": len(feature_descriptions),
        "features": feature_descriptions,
        "algorithm": "K-Means Clustering",
        "num_clusters": 3,
        "encoding": {
            "gender": {"M": 0, "F": 1},
            "discount_usage": {"High": 0, "Low": 1, "Medium": 2},
            "preferred_shopping_time": {"Day": 0, "Night": 1}
        }
    }

@app.get("/model/info", tags=["Information"])
def get_model_information():
    """
    Get model architecture and training information.
    
    Returns:
        Model details and hyperparameters
    """
    
    if kmeans_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "algorithm": "K-Means Clustering",
        "num_clusters": kmeans_model.n_clusters,
        "num_features": kmeans_model.n_features_in_,
        "cluster_centers_shape": kmeans_model.cluster_centers_.shape,
        "inertia": round(float(kmeans_model.inertia_), 2),
        "n_iter": kmeans_model.n_iter_,
        "init_method": "k-means++",
        "random_state": 42,
        "features_used": feature_names if feature_names else "Not available"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print(" Starting Customer Segmentation FastAPI Server")
    print("="*80)
    print(" API Documentation: http://localhost:8000/docs")
    print("Alternative Documentation: http://localhost:8000/redoc")
    print("  OpenAPI Schema: http://localhost:8000/openapi.json")
    print("="*80)
    
    # Check model status
    models_status = {
        "KMeans Model": kmeans_model is not None,
        "Scaler": scaler is not None,
        "Label Encoders": label_encoders is not None,
        "Feature Names": feature_names is not None,
        "Cluster Mapping": cluster_mapping is not None
    }
    
    print("\n Model Status:")
    for name, status in models_status.items():
        status_str = "✓ Loaded" if status else "✗ Missing"
        print(f"  {name}: {status_str}")
    
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
