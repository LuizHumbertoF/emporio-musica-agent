import pandas as pd
import os

def reading_data():
    data_path = "data"
    
    products_path = os.path.join(data_path, "desafio_tecnico_ai_eng - products.csv")
    customers_path = os.path.join(data_path, "desafio_tecnico_ai_eng - customers.csv")
    order_items_path = os.path.join(data_path, "desafio_tecnico_ai_eng - order_items.csv")
    orders_path = os.path.join(data_path, "desafio_tecnico_ai_eng - orders.csv")
    promotions_path = os.path.join(data_path, "desafio_tecnico_ai_eng - promotions.csv")

    all_paths = [products_path, customers_path, order_items_path, orders_path, promotions_path]

    for path in all_paths:
        if os.path.exists(path):
           
            df_path = pd.read_csv(path)
            print(f"'{path}' base succesfuly loaded\n")
            print(df_path.head()) 
        else:
            print(f"File not found in {path}")

        

if __name__ == "__main__":
    reading_data()