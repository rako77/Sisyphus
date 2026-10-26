namespace Shop.Billing;

public static class InvoiceMapper
{
    public static InvoiceDto ToDto(Invoice invoice) => new InvoiceDto
    {
        Id = invoice.Id,
        CustomerId = invoice.CustomerId,
        IssuedAt = invoice.IssuedAt.ToString("O"),
        Total = invoice.Lines.Sum(l => l.Quantity * l.UnitPrice),
        LineCount = invoice.Lines.Count,
        Status = invoice.Status.ToString().ToLowerInvariant(),
    };
}
