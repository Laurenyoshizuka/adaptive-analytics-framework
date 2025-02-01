
    -- models/combined_data_model.sql
    with combined_data as (
        select 
            o.*,
            r."Order ID" as Return_Order_ID, 
            p.Person
        from 
            Orders o
        left join 
            Returns r on o."Order ID" = r."Order ID"
        left join 
            People p on o.Region = p.Region
    )
    select * from combined_data;
    