from abc import ABCMeta, abstractmethod

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class AbstractModel(Base):
    __metaclass__ = ABCMeta
    __abstract__ = True
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass


class UserModel(AbstractModel):
    __tablename__ = "users"

    username = Column(
        String,
        unique=True,
        nullable=False,
    )

    files = relationship(
        "DocumentModel",
        back_populates="user",
    )

    def __init__(self, username):
        self.username = username

    def to_dict(self):
        return {"username": self.username}


class DocumentModel(AbstractModel):
    __tablename__ = "documents"

    file_id = Column(
        String,
        unique=True,
        nullable=False,
    )
    name = Column(
        String,
        nullable=False,
    )
    last_modified = Column(
        DateTime,
        nullable=False,
    )
    provider = Column(
        String,
        nullable=False,
    )
    type = Column(
        String,
        nullable=False,
    )
    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
    )

    user = relationship(
        "UserModel",
        back_populates="files",
    )

    def __init__(self, file_id, name, last_modified, provider, doc_type, user_id):
        self.file_id = file_id
        self.name = name
        self.last_modified = last_modified
        self.provider = provider
        self.type = doc_type
        self.user_id = user_id

    def to_dict(self):
        return {
            "file_id": self.file_id,
            "display_name": self.display_name,
            "last_modified": self.last_modified,
            "user_id": self.user_id,
        }
