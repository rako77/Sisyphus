using Shop.Shared;

namespace Shop.Orders;

public class OrderService
{
    private readonly IOrderRepository _orders;
    private readonly ILogger _log;

    public OrderService(IOrderRepository orders, ILogger log)
    {
        _orders = orders ?? throw new ArgumentNullException(nameof(orders));
        _log = log;
    }

    public decimal CalculateDiscount(Customer customer, decimal subtotal)
    {
        if (subtotal <= 0) return 0m;
        decimal rate = 0m;
        if (customer.IsVip) rate += 0.10m;
        if (customer.YearsActive >= 5) rate += 0.05m;
        else if (customer.YearsActive >= 2) rate += 0.02m;
        if (subtotal > 1000m) rate += 0.03m;
        rate = Math.Min(rate, 0.15m);
        return Math.Round(subtotal * rate, 2, MidpointRounding.AwayFromZero);
    }

    public List<string> Validate(Order order)
    {
        ArgumentNullException.ThrowIfNull(order);
        var errors = new List<string>();
        if (string.IsNullOrWhiteSpace(order.CustomerId))
            errors.Add("Customer is required.");
        if (order.Lines == null || order.Lines.Count == 0)
            errors.Add("Order must have at least one line.");
        else
        {
            foreach (var line in order.Lines)
            {
                if (line.Quantity <= 0) errors.Add($"Line {line.Sku}: quantity must be positive.");
                if (line.UnitPrice < 0) errors.Add($"Line {line.Sku}: price cannot be negative.");
            }
        }
        if (order.DueDate < DateTime.UtcNow.Date)
            errors.Add("Due date cannot be in the past.");
        return errors;
    }

    public decimal ComputeTotal(Order order)
    {
        decimal total = 0;
        foreach (var line in order.Lines)
        {
            if (line.IsCancelled) continue;
            total += line.Quantity * line.UnitPrice;
        }
        return total;
    }

    public bool IsEligibleForFreeShipping(Order order, Customer customer)
    {
        var total = ComputeTotal(order);
        if (customer.IsVip) return true;
        if (order.Lines.Count >= 3 && total >= 50m) return true;
        return total >= 100m && customer.Country == "US";
    }

    public async Task<T> RetryAsync<T>(Func<Task<T>> action, int maxAttempts = 3)
    {
        var delay = TimeSpan.FromMilliseconds(200);
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                return await action().ConfigureAwait(false);
            }
            catch (HttpRequestException ex) when (attempt < maxAttempts && !ex.Message.Contains("404"))
            {
                _log.Warn($"Attempt {attempt} failed: {ex.Message}. Retrying in {delay.TotalMilliseconds}ms");
                await Task.Delay(delay).ConfigureAwait(false);
                delay = TimeSpan.FromMilliseconds(delay.TotalMilliseconds * 2);
            }
        }
    }

    // overload forwarder: noise
    public Task<Order?> GetAsync(string id) => _orders.GetByIdAsync(id);
}
