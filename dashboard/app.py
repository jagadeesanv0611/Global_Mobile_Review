import streamlit as st
import joblib
import os
import pandas as pd

# Title for Sidebar:
st.set_page_config( page_title="Global Mobile Reviews", layout="wide")

# Step 1 - Importing the Load the saved models:
recommend_model = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_4-global_mobile_reviews\models\recommend_model.pkl")
product_df = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_4-global_mobile_reviews\models\product_df.pkl")
scaler = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_4-global_mobile_reviews\models\scaler.pkl")


# Step 2 - Sidebar:
st.sidebar.title("📊 Global Mobile Reviews")
page = st.sidebar.radio( "Navigate", ["Home","EDA","Mobile Recommendation System"] )


# Step 3 - Home page creation:
if page == "Home":
    st.title("Global Mobile Reviews")
    st.set_page_config(page_title="Global Mobile Reviews", layout="wide" )


# Step 4 - EDA page creation:
elif page == "EDA":
     st.subheader("Exploratory Data Analysis")
     st.set_page_config( page_title="EDA", layout="wide" )

     selection = st.pills("Data Visualization", 
                options=["Customer_Sentiment_Distribution"]
                            )
     
     if selection == "Customer_Sentiment_Distribution":
         customer_sentiment_distribution_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_4-global_mobile_reviews\models\customer_sentiment_distribution.pkl")
         st.plotly_chart(customer_sentiment_distribution_fig)




# Step 5 - Recommended products page creation:
elif page == "Mobile Recommendation System":
        st.subheader(" Mobile Recommendation System")
        st.set_page_config(page_title="Mobile Recommendation System", layout="wide" )
        st.sidebar.header("Settings")


# Step 6 - Number of Recommendations:
        top_n = st.sidebar.number_input(
                "Number of Recommendations",
                min_value=1,
                max_value=22,
                value = 5,
                step = 1
            )
        
        col1, col2 = st.columns(2)

        with col1:
            brand_list = sorted(product_df["brand"].unique().tolist())
            selected_brands = st.multiselect("Select Brand(s)", brand_list )

            overall_rating_values = sorted(product_df["rating"].unique().tolist())
            selected_rating = st.select_slider( "Overall Rating", options=overall_rating_values,
                                                value=overall_rating_values[len(overall_rating_values)//2])        

            battery_values = sorted(product_df["battery_life_rating"].unique().tolist())
            selected_battery = st.select_slider( "Battery Rating", options=battery_values,
                                            value=battery_values[len(battery_values)//2])
            
            design_values = sorted(product_df["design_rating"].unique().tolist())
            selected_design = st.select_slider( "Design Rating", options=design_values,
                                            value=design_values[len(design_values)//2])

        with col2:
            selected_price = st.slider("Price ($)", 300, 1500, 800)

            camera_values = sorted(product_df["camera_rating"].unique().tolist())
            selected_camera = st.select_slider( "Camera Rating", options=camera_values,
                                            value=camera_values[len(camera_values)//2])

            performance_values = sorted(product_df["performance_rating"].unique().tolist())
            selected_performance = st.select_slider( "Performance Rating", options=performance_values,
                                            value=performance_values[len(performance_values)//2])
            
            display_values = sorted(product_df["display_rating"].unique().tolist())
            selected_display= st.select_slider( "Display Rating", options=display_values,
                                            value=display_values[len(display_values)//2])
            



# Step 7: Create User Input Vector
        user_input = {
                    "price_usd": selected_price,
                    "rating": selected_rating,
                    "battery_life_rating": selected_battery,
                    "camera_rating": selected_camera,
                    "performance_rating": selected_performance,
                    "design_rating": selected_design,
                    "display_rating": selected_display,

                    "brand_Apple":0,
                    "brand_Google":0,
                    "brand_Motorola":0,
                    "brand_OnePlus":0,
                    "brand_Realme":0,
                    "brand_Samsung":0,
                    "brand_Xiaomi":0
                    }

# Encoding the brand:
        for brand in selected_brands:
            user_input[f"brand_{brand}"] = 1

# convert into dataframe:
        user_df = pd.DataFrame([user_input])


# Step 8: Recommendation Button:
        if st.button("Recommend Mobiles"): 
# Scaling:
            user_scaled = scaler.transform(user_df)

            distances, indices = recommend_model.kneighbors(user_scaled, n_neighbors= int(top_n) )

            recommendations = product_df.iloc[indices[0]].copy()

            recommendations["Similarity (%)"] = ((1 - distances[0]) * 100 ).round(2)


# Step 9: Apply filters:
            recommendations = recommendations[
                                recommendations["brand"].isin(selected_brands)
                                            ]

            recommendations = recommendations[
                                recommendations["price_usd"] <= selected_price
            ]

            recommendations = recommendations[
                recommendations["rating"] <= selected_rating
            ]

            recommendations = recommendations[
                recommendations["battery_life_rating"] <= selected_battery
            ]

            recommendations = recommendations[
                recommendations["camera_rating"] <= selected_camera
            ]

            recommendations = recommendations[
                recommendations["performance_rating"] <= selected_performance
            ]

            recommendations = recommendations[
                recommendations["design_rating"] <= selected_design
            ]

            recommendations = recommendations[
                recommendations["display_rating"] <= selected_display
            ]


# Step 10: Display recommendations:
            if recommendations.empty:

                st.warning("No mobiles found.")

            else:

                st.success(f"{len(recommendations)} mobiles found")

                st.dataframe( recommendations[
                        [
                            "brand",
                            "model",
                            "price_usd",
                            "rating",
                            "battery_life_rating",
                            "camera_rating",
                            "performance_rating",
                            "design_rating",
                            "display_rating",
                            "Similarity (%)"
                        ]
                    ],
                    use_container_width=True
                )



