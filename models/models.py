from sqlalchemy import Integer, String, Boolean, ForeignKey, DateTime, Float, CHAR, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

from pydantic import BaseModel
from database import Base
from typing import List
import datetime

class User(Base):
    __tablename__ = "users"
    user_id : Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_uuid : Mapped[str] = mapped_column(CHAR(36), nullable=False, unique=True)
    user_name : Mapped[str] = mapped_column(String(100), nullable=False)
    user_password : Mapped[str] = mapped_column(String(255), nullable=False)
    phone_number : Mapped[str] = mapped_column(String(15), nullable=False)
    email : Mapped[str] = mapped_column(String(100), nullable=False)
    created_at : Mapped[datetime.datetime] = mapped_column(DateTime, default=func.now())
    last_login : Mapped[datetime.datetime] = mapped_column(DateTime, default=func.now())

    user_body_profile : Mapped["UserBodyProfile"] = relationship(back_populates="user", cascade="all, delete-orphan")
    refresh_tokens: Mapped[List["RefreshToken"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    assistant_threads: Mapped[List["AssistantThread"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    body_measurements_record: Mapped[List["BodyMeasurementRecord"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class UserBodyProfile(Base):
    __tablename__ = "user_body_profile"

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.user_id"), primary_key=True)
    user_age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(10), nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    neck_circumference: Mapped[float] = mapped_column(Float, nullable=False)
    body_fat_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    body_muscle_mass: Mapped[float] = mapped_column(Float, nullable=False)
    body_bone_density: Mapped[float] = mapped_column(Float, nullable=False)


    user: Mapped["User"] = relationship(back_populates="user_body_profile")

class BodyMeasurementRecord(Base):
    __tablename__ = "body_measurements_record"

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.user_id"), primary_key=True)
    recoded_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=func.now())
    left_arm_length: Mapped[float] = mapped_column(Float, nullable=False)
    right_arm_length: Mapped[float] = mapped_column(Float, nullable=False)
    inside_leg_height: Mapped[float] = mapped_column(Float, nullable=False)
    shoulder_to_crotch_height: Mapped[float] = mapped_column(Float, nullable=False)
    shoulder_breadth: Mapped[float] = mapped_column(Float, nullable=False)
    head_circumference: Mapped[float] = mapped_column(Float, nullable=False)
    chest_circumference: Mapped[float] = mapped_column(Float, nullable=False)
    waist_circumference: Mapped[float] = mapped_column(Float, nullable=False)
    hip_circumference: Mapped[float] = mapped_column(Float, nullable=False)
    wrist_right_circumference: Mapped[float] = mapped_column(Float, nullable=False)
    bicep_right_circumference: Mapped[float] = mapped_column(Float, nullable=False)
    forearm_right_circumference: Mapped[float] = mapped_column(Float, nullable=False)
    thigh_left_circumference: Mapped[float] = mapped_column(Float, nullable=False)
    calf_left_circumference: Mapped[float] = mapped_column(Float, nullable=False)
    ankle_left_circumference: Mapped[float] = mapped_column(Float, nullable=False)

    user: Mapped["User"] = relationship(back_populates="body_measurements_record")