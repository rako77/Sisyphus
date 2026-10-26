using System.Data;
using Shop.Shared;

namespace Shop.Orders;

public class OrderRepository : IOrderRepository
{
    private readonly IDbConnectionFactory _db;

    public OrderRepository(IDbConnectionFactory db) => _db = db;

    public async Task<Order?> GetByIdAsync(string id)
    {
        using var conn = await _db.OpenAsync();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "SELECT * FROM Orders WHERE Id = @id AND IsDeleted = 0";
        cmd.Parameters.Add(new SqlParam("@id", id));
        using var reader = await cmd.ExecuteReaderAsync();
        if (!await reader.ReadAsync()) return null;
        return Order.FromReader(reader);
    }

    public async Task<IReadOnlyList<Order>> ListByCustomerAsync(string customerId, int take = 50)
    {
        using var conn = await _db.OpenAsync();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "SELECT TOP (@take) * FROM Orders WHERE CustomerId = @c AND IsDeleted = 0 ORDER BY CreatedAt DESC";
        cmd.Parameters.Add(new SqlParam("@take", take));
        cmd.Parameters.Add(new SqlParam("@c", customerId));
        var results = new List<Order>();
        using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync()) results.Add(Order.FromReader(reader));
        return results;
    }
}
