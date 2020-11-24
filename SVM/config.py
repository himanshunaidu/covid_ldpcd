import mysql.connector
mydb = mysql.connector.connect(
  host="<>",
  user="<>",
  password="<>",
  database="<>"
)

coronet_train_table = 'train_coro_features4'
coronet_test_table = 'test_coro_features4'