from src.predict import predict_transaction

transactions = [
    {"step": 2, "type": "PAYMENT",  "amount": 700,    "nameOrig": "C123", "nameDest": "M2", "oldbalanceOrg": 500000, "oldbalanceDest": 500000},
    {"step": 3, "type": "DEBIT",    "amount": 1000,   "nameOrig": "C123", "nameDest": "M3", "oldbalanceOrg": 500000, "oldbalanceDest": 500000},
    {"step": 4, "type": "PAYMENT",  "amount": 1200,   "nameOrig": "C123", "nameDest": "M4", "oldbalanceOrg": 500000, "oldbalanceDest": 500000},
    {"step": 5, "type": "DEBIT",    "amount": 800,    "nameOrig": "C123", "nameDest": "M5", "oldbalanceOrg": 500000, "oldbalanceDest": 500000},
    {"step": 6, "type": "PAYMENT",  "amount": 600,    "nameOrig": "C123", "nameDest": "M6", "oldbalanceOrg": 500000, "oldbalanceDest": 500000},
    {"step": 7, "type": "PAYMENT",  "amount": 900,    "nameOrig": "C123", "nameDest": "M7", "oldbalanceOrg": 500000, "oldbalanceDest": 500000},
    {"step": 8, "type": "DEBIT",    "amount": 1100,   "nameOrig": "C123", "nameDest": "M8", "oldbalanceOrg": 500000, "oldbalanceDest": 500000},
    {"step": 9, "type": "PAYMENT",  "amount": 750,    "nameOrig": "C123", "nameDest": "M9", "oldbalanceOrg": 500000, "oldbalanceDest": 500000},
    {"step":10, "type": "PAYMENT",  "amount": 650,    "nameOrig": "C123", "nameDest": "M10", "oldbalanceOrg": 500000, "oldbalanceDest": 500000},

    # Suspicious: Large TRANSFER with low balance (high ratio)
    {"step":11, "type": "TRANSFER", "amount": 180000, "nameOrig": "C123", "nameDest": "C999", "oldbalanceOrg": 10000, "oldbalanceDest": 10000},

    # Fraud: Large TRANSFER with low balance
    {"step":12, "type": "TRANSFER", "amount": 900, "nameOrig": "C123", "nameDest": "C888", "oldbalanceOrg": 10000, "oldbalanceDest": 10000},
    
    # Normal: Small DEBIT with normal balance
    {"step":13, "type": "DEBIT", "amount": 200, "nameOrig": "C123", "nameDest": "C777", "oldbalanceOrg": 500000, "oldbalanceDest": 500000},
    
    # Suspicious: Large PAYMENT with low balance
    {"step":14, "type": "PAYMENT",  "amount": 200000, "nameOrig": "C123", "nameDest": "M11", "oldbalanceOrg": 10000, "oldbalanceDest": 10000},
    
    # Suspicious: Large DEBIT with low balance
    {"step":15, "type": "DEBIT",    "amount": 180000, "nameOrig": "C123", "nameDest": "M12", "oldbalanceOrg": 10000, "oldbalanceDest": 10000},
]

print("\n--- ONLINE LEARNING DEMO ---\n")

for i, txn in enumerate(transactions, start=1):
    result = predict_transaction(txn)

    print(f"Txn {i}: {txn['type']} | Amount: {txn['amount']}")
    print(f"→ Global: {result['global_prediction']} | User: {result['user_prediction']} | Final: {result['final_prediction']}")

    print("-" * 50)