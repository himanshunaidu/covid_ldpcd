covid_train_path = '<>'
normal_train_path = '<>'
bacterial_train_path = '<>'
viral_train_path = '<>'

array_write_path = '<>'

covid_ldp_train_path = '<>'
normal_ldp_train_path = '<>'
bacterial_ldp_train_path = '<>'
viral_ldp_train_path = '<>'

covid_test_path = '<>'
normal_test_path = '<>'
bacterial_test_path = '<>'
viral_test_path = '<>'

covid_ldp_test_path = '<>'
normal_ldp_test_path = '<>'
bacterial_ldp_test_path = '<>'
viral_ldp_test_path = '<>'

network_test_save_path = '<>'
network_test_load_path = '<>'

network_load_path = '<>'
network_save_path = '<>'
network_best_load_path = '<>'
network_best_save_path = '<>'

network_ldp_save_path = '<>'
network_ldp_load_path = '<>'
network_ldp_best_load_path = '<>'
network_ldp_best_save_path = '<>'

import mysql.connector
mydb = mysql.connector.connect(
  host="<>",
  user="<>",
  password="<>",
  database="<>"
)

coronet_train_table = 'train_coro_features4'
coronet_test_table = 'test_coro_features4'