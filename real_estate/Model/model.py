import os
import joblib as jb
import streamlit as st
import pandas as pd
import numpy as np



model = jb.load(os.path.expanduser('~/Desktop/Project/real_estate/Model/model.joblib'))
ohe = jb.load(os.path.expanduser('~/Desktop/Project/real_estate/Model/OHE.joblib'))

locations_list = ['1st Block Jayanagar', '1st Block Koramangala', '1st Phase JP Nagar', '2nd Phase Judicial Layout', '2nd Stage Nagarbhavi',
                   '5th Phase JP Nagar', '6th Phase JP Nagar', '7th Phase JP Nagar', '8th Phase JP Nagar', '9th Phase JP Nagar', 'AECS Layout',
                   'Abbigere', 'Akshaya Nagar', 'Ambalipura', 'Ambedkar Nagar', 'Amruthahalli', 'Anandapura', 'Ananth Nagar', 'Anekal', 'Anjanapura',
                   'Ardendale', 'Arekere', 'Attibele', 'BEML Layout', 'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya', 'Badavala Nagar', 'Balagere',
                   'Banashankari', 'Banashankari Stage II', 'Banashankari Stage III', 'Banashankari Stage V', 'Banashankari Stage VI', 'Banaswadi',
                   'Banjara Layout', 'Bannerghatta', 'Bannerghatta Road', 'Basapura', 'Basavangudi', 'Basaveshwara Nagar', 'Battarahalli', 'Begur',
                   'Begur Road', 'Bellandur', 'Benson Town', 'Bharathi Nagar', 'Bhoganhalli', 'Billekahalli', 'Binny Pete', 'Bisuvanahalli',
                   'Bommanahalli', 'Bommasandra', 'Bommasandra Industrial Area', 'Bommenahalli', 'Brookefield', 'Budigere', 'CV Raman Nagar', 
                   'Chamrajpet', 'Chandapura', 'Channasandra', 'Chikka Tirupathi', 'Chikkabanavar', 'Chikkalasandra', 'Choodasandra', 'Cooke Town',
                   'Cox Town', 'Cunningham Road', 'Dairy Circle', 'Dasanapura', 'Dasarahalli', 'Devanahalli', 'Devarachikkanahalli', 'Dodda Nekkundi',
                   'Doddaballapur', 'Doddakallasandra', 'Doddathoguru', 'Domlur', 'Dommasandra', 'EPIP Zone', 'Electronic City', 
                   'Electronic City Phase II', 'Electronics City Phase 1', 'Frazer Town', 'GM Palaya', 'Ganga Nagar', 'Garudachar Palya',
                   'Giri Nagar', 'Gollarapalya Hosahalli', 'Gottigere', 'Green Glen Layout', 'Gubbalala', 'Gunjur', 'Gunjur Palya', 'HAL 2nd Stage',
                   'HBR Layout', 'HRBR Layout', 'HSR Layout', 'Haralur Road', 'Harlur', 'Hebbal', 'Hebbal Kempapura', 'Hegde Nagar', 'Hennur',
                   'Hennur Road', 'Hoodi', 'Horamavu Agara', 'Horamavu Banaswadi', 'Hormavu', 'Hosa Road', 'Hosakerehalli', 'Hoskote', 'Hosur Road',
                   'Hulimavu', 'ISRO Layout', 'ITPL', 'Iblur Village', 'Indira Nagar', 'JP Nagar', 'Jakkur', 'Jalahalli', 'Jalahalli East', 'Jigani', 
                   'Judicial Layout', 'KR Puram', 'Kadubeesanahalli', 'Kadugodi', 'Kaggadasapura', 'Kaggalipura', 'Kaikondrahalli', 'Kalena Agrahara',
                   'Kalkere', 'Kalyan nagar', 'Kambipura', 'Kammanahalli', 'Kammasandra', 'Kanakapura', 'Kanakpura Road', 'Kannamangala', 'Karuna Nagar', 
                   'Kasavanhalli', 'Kasturi Nagar', 'Kathriguppe', 'Kaval Byrasandra', 'Kenchenahalli', 'Kengeri', 'Kengeri Satellite Town', 
                   'Kereguddadahalli', 'Kodichikkanahalli', 'Kodigehaali', 'Kodigehalli', 'Kodihalli', 'Kogilu', 'Konanakunte', 'Koramangala', 'Kothannur',
                   'Kothanur', 'Kudlu', 'Kudlu Gate', 'Kumaraswami Layout', 'Kundalahalli', 'LB Shastri Nagar', 'Laggere', 'Lakshminarayana Pura', 
                   'Lingadheeranahalli', 'Magadi Road', 'Mahadevpura', 'Mahalakshmi Layout', 'Mallasandra', 'Malleshpalya', 'Malleshwaram', 'Marathahalli',
                   'Margondanahalli', 'Marsur', 'Mico Layout', 'Munnekollal', 'Murugeshpalya', 'Mysore Road', 'NGR Layout', 'NRI Layout', 'Naganathapura',
                   'Nagappa Reddy Layout', 'Nagarbhavi', 'Nagasandra', 'Nagavara', 'Nagavarapalya', 'Narayanapura', 'Neeladri Nagar', 'OMBR Layout',
                   'Old Airport Road', 'Old Madras Road', 'Others', 'Padmanabhanagar', 'Pai Layout', 'Panathur', 'Parappana Agrahara',
                   'Pattandur Agrahara', 'Poorna Pragna Layout', 'Prithvi Layout', 'R.T. Nagar', 'Rachenahalli', 'Raja Rajeshwari Nagar', 'Rajaji Nagar',
                   'Rajiv Nagar', 'Ramagondanahalli', 'Ramamurthy Nagar', 'Rayasandra', 'Sahakara Nagar', 'Sanjay nagar', 'Sarakki Nagar', 'Sarjapur',
                   'Sarjapur  Road', 'Sarjapura - Attibele Road', 'Sector 2 HSR Layout', 'Sector 7 HSR Layout', 'Seegehalli', 'Shampura', 'Shivaji Nagar',
                   'Singasandra', 'Somasundara Palya', 'Sompura', 'Sonnenahalli', 'Subramanyapura', 'Sultan Palaya', 'TC Palaya', 'Talaghattapura', 
                   'Thanisandra', 'Thigalarapalya', 'Thubarahalli', 'Thyagaraja Nagar', 'Tindlu', 'Tumkur Road', 'Ulsoor', 'Uttarahalli', 'Varthur',
                   'Varthur Road', 'Vasanthapura', 'Vidyaranyapura', 'Vijayanagar', 'Vishveshwarya Layout', 'Vishwapriya Layout', 'Vittasandra', 
                   'Whitefield', 'Yelachenahalli', 'Yelahanka', 'Yelahanka New Town', 'Yelenahalli', 'Yeshwanthpur']


st.title("Bangalore House Price Predictor")


selected_location = st.selectbox("Select Location", sorted(locations_list))
total_sqft = st.number_input("Total Square Feet", min_value=300, max_value=50000, value=1200)
bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

if st.button("Predict Price"):
    num_data = np.array([[bhk, total_sqft, bath]])

    loc_encoded = ohe.transform([[selected_location]])
    if hasattr(loc_encoded, 'toarray'):
        loc_encoded = loc_encoded.toarray()
    
    # 3. Combine Features (Must match your training column order!)
    # Most common order: [Numerical Features, Categorical Features]
    final_features = np.hstack([num_data, loc_encoded])
    
    # 4. Predict & Back-transform from log1p
    log_prediction = model.predict(final_features)
    final_price = np.expm1(log_prediction)[0]
    
    st.success(f"The estimated price is ₹{final_price:.2f} Lakhs")
