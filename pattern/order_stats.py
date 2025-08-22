import numpy as np
import pandas as pd
import utl
import datetime

class OrderStats:
    def __init__(self, load_last_data = False, filepath="", num_rows = 1000):
        if load_last_data == False:
            self.num_rows = num_rows 
            self.read_gzip(filepath)
            self.min_threshold = 0.1
            
    def read_gzip(self, file_path):

        df = pd.read_csv(file_path, compression="gzip",sep= ";",encoding="utf-8",keep_default_na=False, dtype=str, low_memory=False ,  nrows=self.num_rows  ,
                 usecols=[
                     "PartDesc1",
                        "PartDesc2",
                        "PartDesc3",
                        "PartDesc4", 
                        "ResCapacity(h)",
                        "ProdOrderStorageArea",
                        "ProdOrderCreationDate",
                        "ProdOrderParentOrderOID",
                        #"ProdOrderStatus", 
                        "PartID",
                        "PartGroup",
                        "PartType",
                        "PartTypeDescription",
                        "PartVariant",
                        "PartVariantDesc",
                        "ProdOrderStartDate",
                        "ProdOrderEndDate",
                        "ProdOrderPlanDate",
                        "ProdOrderReqStartDate",
                        "ProdOrderFinishedQty",
                        "ProdOrderTargetQty",
                        "ProdOrderCycleTime",
                        "ProdOrderOptimalLotSize",
                        "ProdOrderOID",
                        "ProdOrderComplDate",
                        "ProdOrderLineStatusDesc",
                        "ProdOrderLineStorageArea",
                        "ProdOrderLineTargetDate",
                        "ProdOrderActCreationDate",
                        "ProdOrderActOperation",
                        "ProdOrderActOID",
                        "ProdOrderActProductionQty",
                        "ProdOrderActActualSetupTime",
                        "ProdOrderActActualUnitTime",
                        "ProdOrderActTargetUnitTime",
                        "ProdOrderActTargetSetupTime",
                        "ProdOrderActEndDate",
                        "ProdOrderActStartDate",
                        "ProdOrderActEndTime",
                        "ProdOrderActStartTime",
                        "ProdOrderActReportedQty",
                        "OrderDocLineCreationDate",
                        "OrderDocLineStorageArea",
                        "OrderDocLineRequestedDate",
                        "OrderDocLineDeliveryDate",
                        #"OrderDocLineOID",
                        #"OrderDocLineOpen",
                        "PurOrderLineCreationDate",
                        "PurOrderDocDate",
                        "PurOrderLineQty",
                        "PurOrderLineDeliveryTime",
                        "PurOrderLineStorageArea",
                        "Company",
                        "ProcessDate"
                        #"PurOrderLineOpen",
                        #"PurOrderSupplierNo"
                        ]) #.dropna(subset=[
                        #"PartID" 
                        #                  ])
                # Speichern der ersten 100 Zeilen in eine CSV-Datei
        output_file = "../tests/test_data/output/DocData.csv"
        time_stamp_file = utl.add_timestamp_to_filename(output_file)
        df.to_csv(time_stamp_file, index=False, sep=";")
        self.data = df
        self.part_ids = list(self.data["PartID"])  # Liste der PartIDs für den Index-Zugriff
        print(f"gespeichert in: {time_stamp_file}")  


