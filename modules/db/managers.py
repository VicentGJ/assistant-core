from modules.db.interfaces import DatabaseInterface
from modules.db.models import DocumentModel, UserModel


# TODO: Error handling
class DatabaseManager:
    def __init__(self, db: DatabaseInterface):
        self.db = db

    def register_user(self, username: str) -> None:
        try:
            user = self.db.read(UserModel, {"username": username})
            if user:
                print(f"CHECKED: user {username} is in database.")
            else:
                user = UserModel(username=username)
                self.db.create(user)
                print("" f"CREATED: user {username} in database.")
        except Exception as e:
            raise Exception(f"Error registering user {username}: {e}")

    def get_document(self, file_id):
        doc_list = self.db.read(DocumentModel, {"file_id": file_id})
        if doc_list and len(doc_list) == 1:
            return doc_list[0]
        else:
            return None

    def create_document(
        self, username, file_id, name, last_modified, provider, doc_type
    ):
        [user] = self.db.read(UserModel, {"username": username})
        if user:
            new_doc = DocumentModel(
                file_id=file_id,
                name=name,
                last_modified=last_modified,
                provider=provider,
                doc_type=doc_type,
                user_id=user.id,
            )
            self.db.create(new_doc)
            print(f"CREATED: {new_doc.name}")
        else:
            print(f"User {username} not found in database.")

    def update_modified_document(self, file_id, last_modified):
        return self.db.update(
            DocumentModel,
            {"file_id": file_id},
            {"last_modified": last_modified},
        )

    def register_document(
        self, username, file_id, name, last_modified, provider, doc_type
    ):
        [user] = self.db.read(UserModel, {"username": username})
        if user:
            doc = self.db.read(DocumentModel, {"file_id": file_id})
            if doc:
                self.update_modified_document(file_id, last_modified)
                print(f"UPDATED: {doc[0].name}")
            else:
                self.create_document(
                    username, file_id, name, last_modified, provider, doc_type
                )
        else:
            print(f"User {username} not found in database.")

    def delete_all_documents_by_username(self, username):
        [user] = self.db.read(UserModel, {"username": username})
        if user:
            self.db.delete(DocumentModel, {"user_id": user.id})
            print(f"DELETED: all documents of user {username} in database.")
        else:
            print(f"User {username} not found in database.")
