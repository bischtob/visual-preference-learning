from server import db

meta = db.metadata
for table in reversed(meta.sorted_tables):
    db.session.execute(table.delete())

db.session.commit()
