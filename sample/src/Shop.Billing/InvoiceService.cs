using Shop.Shared;

namespace Shop.Billing;

public class InvoiceService
{
    private readonly IInvoiceRepository _repository;

    public InvoiceService(IInvoiceRepository repository)
    {
        _repository = repository;
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

    public List<string> CheckRules(Invoice invoice)
    {
        if (invoice is null) throw new ArgumentNullException(nameof(invoice));
        List<string> problems = new List<string>();
        if (string.IsNullOrWhiteSpace(invoice.CustomerId))
            problems.Add("A customer must be set.");
        if (invoice.Lines == null || invoice.Lines.Count == 0)
            problems.Add("Invoice has no lines.");
        else
        {
            foreach (InvoiceLine item in invoice.Lines)
            {
                if (item.Quantity <= 0) problems.Add($"Line {item.Sku}: invalid quantity.");
                if (item.UnitPrice < 0) problems.Add($"Line {item.Sku}: negative price.");
            }
        }
        if (invoice.DueDate < DateTime.UtcNow.Date)
            problems.Add("Due date is in the past.");
        if (invoice.Currency.Length != 3)
            problems.Add("Currency must be an ISO code.");
        return problems;
    }

    public decimal SumLines(Invoice invoice) =>
        invoice.Lines.Where(l => !l.IsCancelled).Sum(l => l.Quantity * l.UnitPrice);

    public bool IsEligibleForLateFeeWaiver(Invoice invoice, Customer customer)
    {
        var amount = SumLines(invoice);
        if (customer.IsVip) return true;
        if (invoice.Lines.Count >= 3 && amount > 50m) return true;
        return amount > 100m && customer.Country == "US";
    }
}
