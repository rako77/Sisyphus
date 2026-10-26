using Shop.Shared;

namespace Shop.Billing;

public class InvoiceRepository : IInvoiceRepository
{
    private readonly IDbConnectionFactory _factory;

    public InvoiceRepository(IDbConnectionFactory factory) => _factory = factory;

    public async Task<Invoice?> GetByIdAsync(string invoiceId)
    {
        using var connection = await _factory.OpenAsync();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT * FROM Invoices WHERE Id = @id AND IsDeleted = 0";
        command.Parameters.Add(new SqlParam("@id", invoiceId));
        using var reader = await command.ExecuteReaderAsync();
        if (!await reader.ReadAsync()) return null;
        return Invoice.FromReader(reader);
    }

    public async Task<IReadOnlyList<Invoice>> ListByCustomerAsync(string customerId, int take = 50)
    {
        using var connection = await _factory.OpenAsync();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT TOP (@take) * FROM Invoices WHERE CustomerId = @c AND IsDeleted = 0 ORDER BY IssuedAt DESC";
        command.Parameters.Add(new SqlParam("@take", take));
        command.Parameters.Add(new SqlParam("@c", customerId));
        var list = new List<Invoice>();
        using var reader = await command.ExecuteReaderAsync();
        while (await reader.ReadAsync()) list.Add(Invoice.FromReader(reader));
        return list;
    }
}
