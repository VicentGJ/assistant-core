from abc import ABC, abstractmethod
from settings import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from modules.db.models import AbstractModel, Base


# TODO: Error handling
class DatabaseInterface(ABC):
    @abstractmethod
    def create(self, model):
        pass

    @abstractmethod
    def read(self, model, filter_by=None):
        pass

    @abstractmethod
    def update(self, model, filter_by, update_values):
        pass

    @abstractmethod
    def delete(self, model, filter_by):
        pass


class SQLAlchemyDatabase(DatabaseInterface):
    def __init__(self):
        try:
            self.engine = create_engine(
                settings.db_url + "?sslmode=disable", pool_size=10, max_overflow=20
            )
            self.session_local = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
            Base.metadata.create_all(
                bind=self.engine
            )  # TODO: This is not working properly with Supabase
        except Exception as e:
            raise Exception(f"An error occurred creating database: {e}")

    def _get_db(self):
        db = self.session_local()
        try:
            yield db
        except Exception as e:
            raise Exception(f"Error during session: {e}")
        finally:
            db.close()

    def create(self, model: AbstractModel) -> AbstractModel:
        try:
            db: Session = next(self._get_db())
            db.add(model)
            db.commit()
            db.refresh(model)
            return model
        except Exception as e:
            raise Exception(f"An error occurred creating object: {e}")

    def read(
        self, model: AbstractModel, filter_by: dict | None = None
    ) -> list[AbstractModel]:
        try:
            db: Session = next(self._get_db())
            if filter_by:
                objs: list[AbstractModel] = db.query(model).filter_by(**filter_by).all()
            else:
                objs = db.query(model).all()
            return objs
        except Exception as e:
            raise Exception(f"An error occurred reading object: {e}")

    def update(
        self, model: AbstractModel, filter_by: dict, update_values: dict
    ) -> list[AbstractModel]:
        try:
            db: Session = next(self._get_db())
            objs: list[AbstractModel] = db.query(model).filter_by(**filter_by).all()
            for obj in objs:
                for key, value in update_values.items():
                    setattr(obj, key, value)
            db.commit()
            return objs
        except Exception as e:
            raise Exception(f"An error occurred updating object: {e}")

    def delete(self, model: AbstractModel, filter_by: dict) -> list[AbstractModel]:
        try:
            db: Session = next(self._get_db())
            objs: list[AbstractModel] = db.query(model).filter_by(**filter_by).all()
            for obj in objs:
                db.delete(obj)
            db.commit()
            return objs
        except Exception as e:
            raise Exception(f"An error occurred deleting object: {e}")
