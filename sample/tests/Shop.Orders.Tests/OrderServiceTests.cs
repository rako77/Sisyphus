namespace Shop.Orders.Tests;

public class OrderServiceTests
{
    [Fact]
    public void Discount_for_vip_is_capped()
    {
        var svc = new OrderService(new FakeRepo(), new NullLogger());
        var customer = new Customer("1", "A", "US") { IsVip = true, YearsActive = 10 };
        var discount = svc.CalculateDiscount(customer, 2000m);
        Assert.Equal(300m, discount);
    }

    [Fact]
    public void Discount_for_new_customer_is_zero()
    {
        var svc = new OrderService(new FakeRepo(), new NullLogger());
        var customer = new Customer("2", "B", "US") { IsVip = false, YearsActive = 0 };
        var discount = svc.CalculateDiscount(customer, 200m);
        Assert.Equal(0m, discount);
    }
}
