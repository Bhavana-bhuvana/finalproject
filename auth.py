
import bcrypt
from pymongo import MongoClient
import random
import streamlit as st
from utils import send_otp_email
import login_signup
MONGO_URI = "mongodb+srv://komal0mallaram:Qwerty%401234@cluster0.lhfjf0c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["New_DB"]
users_collection = db["Users"]
pending_users_collection = db["pending_users"]

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def send_otp(email, full_name=None, username=None, purpose="signup"):
    import random

    otp = str(random.randint(100000, 999999))

    if purpose == "signup":
        # check if already signed up
        if users_collection.find_one({"email": email}):
            return None  # signup not allowed

        data = {
            "email": email,
            "full_name": full_name,
            "username": username,
            "otp": otp,
            "otp_verified": False,
            "purpose": "signup"
        }

    elif purpose == "reset":
        # don't add name or username
        if not users_collection.find_one({"email": email}):
            return None  # can't reset non-existing user

        data = {
            "email": email,
            "otp": otp,
            "otp_verified": False,
            "purpose": "reset"
        }

    pending_users_collection.update_one(
        {"email": email, "purpose": purpose},
        {"$set": data},
        upsert=True
    )

    return otp


def signup_user(full_name, email, password):
    # user = pending_users_collection.find_one({"email": email})
    # if not user or not user.get("otp_verified"):
    #     return False, "OTP not verified or email not found."

    # Now move to main users collection
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        "full_name": full_name,
        
        "email": email,
        "password": hashed_pw
    })
    # Clean up pending user
    return True, "Signup successful!"

def login_user(identifier, password):
    user = users_collection.find_one({
        "$or": [{"email": identifier}, {"username": identifier}]
    })
    if not user:
        return False, "User not found."
    if not check_password(password, user["password"]):
        return False, "Incorrect password." 
    return True, "Login successful."
def verify_otp(email, otp, purpose="signup"):

    record = pending_users_collection.find_one({"email": email, "purpose": purpose})
    if record and record["otp"] == otp:
        pending_users_collection.update_one(
            {"email": email, "purpose": purpose},
            {"$set": {"otp_verified": True}}
        )
        return True
    return False

def reset_password(email, new_password):
    hashed = hash_password(new_password)
    users_collection.update_one(
        {"email": email},
        {"$set": {"password": hashed}}
    )
    return True

def mark_user_verified(email,purpose="signup"):
    pending_users_collection.delete_one({"email": email, "purpose": purpose})



def get_user_by_email_or_username(identifier):
    return users_collection.find_one({
        "$or": [{"email": identifier}, {"username": identifier}]
    })
# *************************************************************************************
# import bcrypt
# import jwt
# import datetime
# from pymongo import MongoClient
# import random

# MONGO_URI = "mongodb+srv://komal0mallaram:Qwerty%401234@cluster0.lhfjf0c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# client = MongoClient(MONGO_URI)
# db = client["New_DB"]
# users_collection = db["Users"]
# pending_users_collection = db["pending_users"]

# SECRET_KEY = "YOUR_SECRET_KEY_CHANGE_THIS_TO_ENV_VAR"  # Replace with env var in prod

# def hash_password(password):
#     return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

# def check_password(password, hashed):
#     return bcrypt.checkpw(password.encode(), hashed)

# def generate_token(user):
#     payload = {
#         "email": user["email"],
#         "username": user["username"],
#         "user_id": user.get("user_id"),
#         "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7),
#     }
#     token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
#     return token

# def get_user_from_token(token):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
#         email = payload.get("email")
#         return users_collection.find_one({"email": email})
#     except Exception:
#         return None

# def send_otp(email, full_name=None, username=None, purpose="signup"):
#     otp = str(random.randint(100000, 999999))

#     if purpose == "signup":
#         if users_collection.find_one({"email": email}):
#             return None  # signup not allowed (email exists)
#         data = {
#             "email": email,
#             "full_name": full_name,
#             "username": username,
#             "otp": otp,
#             "otp_verified": False,
#             "purpose": "signup"
#         }

#     elif purpose == "reset":
#         if not users_collection.find_one({"email": email}):
#             return None  # can't reset non-existing user
#         data = {
#             "email": email,
#             "otp": otp,
#             "otp_verified": False,
#             "purpose": "reset"
#         }

#     pending_users_collection.update_one(
#         {"email": email, "purpose": purpose},
#         {"$set": data},
#         upsert=True
#     )

#     return otp

# def generate_user_id_from_email(email):
#     # Extract part before '@'
#     user_id = email.split('@')[0]
#     # Optional: sanitize or ensure uniqueness here if needed
#     return user_id

# def signup_user(full_name, username, email, password):
#     user = pending_users_collection.find_one({"email": email, "purpose": "signup"})
#     if not user or not user.get("otp_verified"):
#         return False, "OTP not verified or email not found."

#     # Generate user_id automatically from email
#     user_id = generate_user_id_from_email(email)

#     hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
#     users_collection.insert_one({
#         "full_name": full_name,
#         "username": username,
#         "email": email,
#         "password": hashed_pw,
#         "user_id": user_id
#     })
#     pending_users_collection.delete_one({"email": email, "purpose": "signup"})
#     return True, "Signup successful!"

# def login_user(identifier, password):
#     user = users_collection.find_one({
#         "$or": [{"email": identifier}, {"username": identifier}]
#     })
#     if not user:
#         return False, "User not found.", None
#     if not check_password(password, user["password"]):
#         return False, "Incorrect password.", None
#     token = generate_token(user)
#     return True, "Login successful.", token

# def reset_password(email, new_password):
#     hashed = hash_password(new_password)
#     users_collection.update_one(
#         {"email": email},
#         {"$set": {"password": hashed}}
#     )
#     pending_users_collection.delete_one({"email": email, "purpose": "reset"})
#     return True

# def mark_user_verified(email, purpose="signup"):
#     pending_users_collection.delete_one({"email": email, "purpose": purpose})

# def get_user_by_email_or_username(identifier):
#     return users_collection.find_one({
#         "$or": [{"email": identifier}, {"username": identifier}]
#     })
