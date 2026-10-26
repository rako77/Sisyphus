namespace Shop.Orders;

public static class OrderMapper
{
    public static OrderDto ToDto(Order order)
    {
        return new OrderDto
        {
            Id = order.Id,
            CustomerId = order.CustomerId,
            CreatedAt = order.CreatedAt.ToString("O"),
            Total = order.Lines.Sum(l => l.Quantity * l.UnitPrice),
            LineCount = order.Lines.Count,
            Status = order.Status.ToString().ToLowerInvariant(),
        };
    }
}
