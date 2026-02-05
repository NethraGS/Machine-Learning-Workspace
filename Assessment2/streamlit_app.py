import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODELS AND DATA (No caching to avoid hashing issues)
# ============================================================================

@st.cache_resource
def load_models():
    """Load saved models and preprocessors"""
    with open('customer_cluster_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    
    with open('cluster_mapping.pkl', 'rb') as f:
        cluster_mapping = pickle.load(f)
    
    return kmeans, scaler, encoders, features, cluster_mapping

def load_and_process_data():
    """Load and process the dataset with clusters"""
    # Read and strip quotes from each line
    with open('CustomerData.csv', 'r') as f:
        lines = [line.strip().strip('"') for line in f.readlines()]
    
    csv_string = '\n'.join(lines)
    df = pd.read_csv(StringIO(csv_string))
    df.rename(columns={df.columns[0]: 'CustomerID'}, inplace=True)
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

# Load models
kmeans, scaler, encoders, features, cluster_mapping = load_models()

# Load data
df_original = load_and_process_data()

# Add clusters to data
def add_clusters(df, kmeans, scaler, encoders):
    """Add cluster predictions to the dataset"""
    df_with_clusters = df.copy()
    
    try:
        # Encode categorical variables
        df_with_clusters['Gender_Encoded'] = encoders['le_gender'].transform(df_with_clusters['Gender'])
        df_with_clusters['DiscountUsage_Encoded'] = encoders['le_discount'].transform(df_with_clusters['DiscountUsage'])
        df_with_clusters['PreferredShoppingTime_Encoded'] = encoders['le_time'].transform(df_with_clusters['PreferredShoppingTime'])
        
        # Prepare feature matrix
        feature_cols = ['Age', 'Gender_Encoded', 'AnnualIncome', 'TotalSpent', 'AvgOrderValue', 
                       'MonthlyPurchases', 'DiscountUsage_Encoded', 'AppTimeMinutes', 
                       'PreferredShoppingTime_Encoded']
        X = df_with_clusters[feature_cols]
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        df_with_clusters['Cluster'] = kmeans.predict(X_scaled)
    except Exception as e:
        st.warning(f"Could not add clusters: {e}")
        df_with_clusters['Cluster'] = 0
    
    return df_with_clusters

df_original = add_clusters(df_original, kmeans, scaler, encoders)

# Cluster descriptions
cluster_info = {
    0: {
        "name": "🏆 High-Value Loyal Customers",
        "description": "Premium customers with exceptional spending power and engagement",
        "characteristics": [
            "✓ Highest average spending",
            "✓ Strong app engagement",
            "✓ High-value purchases",
            "✓ Loyal & consistent buyers"
        ],
        "strategy": "Premium support, exclusive offers, VIP programs"
    },
    1: {
        "name": "💼 Value-Seeking Regular Customers",
        "description": "Regular customers with moderate spending who seek good value",
        "characteristics": [
            "✓ Consistent purchase frequency",
            "✓ Moderate spending levels",
            "✓ Balanced engagement",
            "✓ Price-value conscious"
        ],
        "strategy": "Loyalty rewards, bundle deals, seasonal promotions"
    },
    2: {
        "name": "🎯 Price-Sensitive Occasional Customers",
        "description": "Occasional buyers with lower engagement and spending",
        "characteristics": [
            "✓ Low app usage",
            "✓ Minimal spending",
            "✓ Deal-driven purchases",
            "✓ Sporadic engagement"
        ],
        "strategy": "Discount campaigns, win-back offers, introductory deals"
    }
}

# Title and header
st.markdown("# 🎯 Customer Segmentation & Clustering")
st.markdown("Predict which customer segment a new customer belongs to using AI-powered K-Means clustering")
st.markdown("---")

# Create two main sections: Prediction and Analytics
tab1, tab2, tab3 = st.tabs(["🔮 Predict Cluster", "📊 Analytics Dashboard", "📈 Model Insights"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.subheader("Customer Information Form")
    
    # Create columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Personal Information")
        age = st.slider("Age", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", options=['M', 'F'], index=0, key="gender_select")
        
        st.markdown("### Financial Information")
        annual_income = st.number_input("Annual Income (₹)", min_value=10000, max_value=500000, value=80000, step=5000)
        total_spent = st.number_input("Total Spent (₹)", min_value=0, max_value=100000, value=15000, step=500)
    
    with col2:
        st.markdown("### Purchase Behavior")
        avg_order_value = st.slider("Average Order Value (₹)", min_value=100, max_value=10000, value=1500, step=100)
        monthly_purchases = st.slider("Monthly Purchases", min_value=1, max_value=100, value=8, step=1)
        
        st.markdown("### Preferences")
        discount_usage = st.selectbox("Discount Usage", options=['Low', 'Medium', 'High'], index=1, key="discount_select")
        app_time = st.slider("App Time (minutes/month)", min_value=0, max_value=1000, value=120, step=10)
        shopping_time = st.selectbox("Preferred Shopping Time", options=['Day', 'Night'], index=0, key="time_select")
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button(" Predict Cluster", use_container_width=True, type="primary")
    
    if predict_button:
        # Encode categorical variables
        gender_encoded = encoders['le_gender'].transform([gender])[0]
        discount_encoded = encoders['le_discount'].transform([discount_usage])[0]
        time_encoded = encoders['le_time'].transform([shopping_time])[0]
        
        # Create feature vector
        features_input = np.array([[
            age, gender_encoded, annual_income, total_spent, avg_order_value,
            monthly_purchases, discount_encoded, app_time, time_encoded
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features_input)
        cluster_pred = kmeans.predict(features_scaled)[0]
        
        # Calculate confidence
        distances = np.linalg.norm(features_scaled - kmeans.cluster_centers_, axis=1)
        confidence = (1 - distances[cluster_pred] / np.max(distances)) * 100
        
        st.success("Prediction Complete!")
        st.markdown("---")
        
        # Display key metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Assigned Cluster", f"Cluster {cluster_pred}", delta="Prediction", delta_color="off")
        
        with metric_col2:
            st.metric("Confidence", f"{confidence:.1f}%", delta="Certainty", delta_color="off")
        
        with metric_col3:
            st.metric("Total Spent", f"₹{total_spent:,.0f}", delta="Lifetime Value", delta_color="off")
        
        with metric_col4:
            st.metric("Annual Income", f"₹{annual_income:,.0f}", delta="Financial Capacity", delta_color="off")
        
        st.markdown("---")
        
        # Cluster details
        cluster_num = int(cluster_pred)
        cluster_details = cluster_info[cluster_num]
        
        st.markdown(f"### {cluster_details['name']}")
        st.info(cluster_details['description'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Key Characteristics")
            for char in cluster_details['characteristics']:
                st.write(char)
        
        with col2:
            st.markdown("#### Recommended Strategy")
            st.write(cluster_details['strategy'])
        
        st.markdown("---")
        
        # Customer profile comparison
        st.subheader(" Customer Profile Analysis")
        
        profile_col1, profile_col2, profile_col3 = st.columns(3)
        
        with profile_col1:
            st.markdown("#### Input Profile")
            profile_data = {
                "Age": age,
                "Gender": gender,
                "Annual Income": f"₹{annual_income:,}",
                "Total Spent": f"₹{total_spent:,}",
                "Avg Order Value": f"₹{avg_order_value:,}",
                "Monthly Purchases": monthly_purchases,
                "Discount Usage": discount_usage,
                "App Time": f"{app_time} min",
                "Shopping Time": shopping_time
            }
            for key, value in profile_data.items():
                st.write(f"**{key}:** {value}")
        
        with profile_col2:
            st.markdown("#### Cluster Averages")
            # Calculate cluster stats from original data (after loading)
            cluster_stats = df_original.groupby('Cluster').agg({
                'Age': 'mean',
                'AnnualIncome': 'mean',
                'TotalSpent': 'mean',
                'AvgOrderValue': 'mean',
                'MonthlyPurchases': 'mean',
                'AppTimeMinutes': 'mean'
            }).round(0)
            
            if cluster_num in cluster_stats.index:
                stats = cluster_stats.loc[cluster_num]
                st.write(f"**Age:** {stats['Age']:.0f} years")
                st.write(f"**Annual Income:** ₹{stats['AnnualIncome']:,.0f}")
                st.write(f"**Total Spent:** ₹{stats['TotalSpent']:,.0f}")
                st.write(f"**Avg Order Value:** ₹{stats['AvgOrderValue']:,.0f}")
                st.write(f"**Monthly Purchases:** {stats['MonthlyPurchases']:.0f}")
                st.write(f"**App Time:** {stats['AppTimeMinutes']:.0f} min")
        
        with profile_col3:
            st.markdown("#### Deviation from Cluster")
            cluster_stats = df_original.groupby('Cluster').agg({
                'Age': 'mean',
                'AnnualIncome': 'mean',
                'TotalSpent': 'mean'
            }).round(0)
            
            if cluster_num in cluster_stats.index:
                avg_age = cluster_stats.loc[cluster_num, 'Age']
                avg_income = cluster_stats.loc[cluster_num, 'AnnualIncome']
                avg_spent = cluster_stats.loc[cluster_num, 'TotalSpent']
                
                age_dev = ((age - avg_age) / avg_age * 100) if avg_age != 0 else 0
                income_dev = ((annual_income - avg_income) / avg_income * 100) if avg_income != 0 else 0
                spent_dev = ((total_spent - avg_spent) / avg_spent * 100) if avg_spent != 0 else 0
                
                st.write(f"**Age:** {age_dev:+.1f}%")
                st.write(f"**Income:** {income_dev:+.1f}%")
                st.write(f"**Spending:** {spent_dev:+.1f}%")

with tab2:
    st.subheader("Customer Segmentation Analytics")
    
    try:
        # Load data for visualization
        df = df_original.copy()
        
        # Ensure Cluster column exists
        if 'Cluster' not in df.columns:
            st.error("Cluster column not found in data. Please refresh the page.")
        else:
            # Calculate cluster statistics
            cluster_stats = df.groupby('Cluster').agg({
                'Age': 'mean',
                'AnnualIncome': 'mean',
                'TotalSpent': 'mean',
                'AvgOrderValue': 'mean',
                'MonthlyPurchases': 'mean',
                'AppTimeMinutes': 'mean',
                'CustomerID': 'count'
            }).round(2)
            
            cluster_stats.columns = ['Avg Age', 'Avg Income', 'Avg Spent', 'Avg Order Value', 'Avg Monthly Purchases', 'Avg App Time', 'Count']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_customers = len(df)
                st.metric("Total Customers", f"{total_customers:,}")
            
            with col2:
                total_clusters = df['Cluster'].nunique()
                st.metric("Total Clusters", total_clusters)
            
            with col3:
                avg_spending = df['TotalSpent'].mean()
                st.metric("Avg Spending", f"₹{avg_spending:,.0f}")
            
            with col4:
                avg_income = df['AnnualIncome'].mean()
                st.metric("Avg Annual Income", f"₹{avg_income:,.0f}")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Customer Distribution by Cluster")
                cluster_counts = df['Cluster'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black', linewidth=2)
                ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
                ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
                ax.set_title('Customer Count by Cluster', fontsize=14, fontweight='bold')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig, use_container_width=True)
                plt.close()
            
            with col2:
                st.markdown("#### Average Spending by Cluster")
                avg_spending_by_cluster = df.groupby('Cluster')['TotalSpent'].mean().sort_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(avg_spending_by_cluster.index, avg_spending_by_cluster.values, color=colors, edgecolor='black', linewidth=2)
                ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
                ax.set_ylabel('Average Total Spent (₹)', fontsize=12, fontweight='bold')
                ax.set_title('Average Spending by Cluster', fontsize=14, fontweight='bold')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'₹{int(height):,}',
                            ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig, use_container_width=True)
                plt.close()
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Cluster Statistics")
                st.dataframe(cluster_stats, use_container_width=True)
            
            with col2:
                st.markdown("#### Income vs Spending Scatter")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for cluster in sorted(df['Cluster'].unique()):
                    cluster_data = df[df['Cluster'] == cluster]
                    ax.scatter(cluster_data['AnnualIncome'], cluster_data['TotalSpent'], 
                              label=f'Cluster {cluster}', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
                
                ax.set_xlabel('Annual Income (₹)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Total Spent (₹)', fontsize=12, fontweight='bold')
                ax.set_title('Income vs Spending by Cluster', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig, use_container_width=True)
                plt.close()
    
    except Exception as e:
        st.error(f"Error loading analytics dashboard: {str(e)}")
        st.info("Please ensure all data files are loaded correctly.")


with tab3:
    st.subheader("Model Architecture & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Information")
        st.write("""
        **Algorithm:** K-Means Clustering
        **Number of Clusters:** 3
        **Features Used:** 9
        **Training Samples:** ~100
        **Random State:** 42
        **Initialization:** k-means++
        """)
    
    with col2:
        st.markdown("#### ✨ Features")
        feature_list = [
            "Age",
            "Gender (Encoded)",
            "Annual Income",
            "Total Spent",
            "Average Order Value",
            "Monthly Purchases",
            "Discount Usage (Encoded)",
            "App Time (minutes)",
            "Preferred Shopping Time (Encoded)"
        ]
        for i, feat in enumerate(feature_list, 1):
            st.write(f"{i}. {feat}")
    
    st.markdown("---")
    
    st.markdown("#### 🎯 Cluster Profiles")
    
    for cluster_id in [0, 1, 2]:
        with st.expander(f"**{cluster_info[cluster_id]['name']}**", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Description:** {cluster_info[cluster_id]['description']}")
                st.write("**Characteristics:**")
                for char in cluster_info[cluster_id]['characteristics']:
                    st.write(char)
            
            with col2:
                st.write("**Marketing Strategy:**")
                st.write(cluster_info[cluster_id]['strategy'])
    
    st.markdown("---")

