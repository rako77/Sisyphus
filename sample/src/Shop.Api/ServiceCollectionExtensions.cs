namespace Shop.Api;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddShop(this IServiceCollection services)
    {
        services.AddSingleton<IDbConnectionFactory, SqlConnectionFactory>();
        services.AddScoped<IOrderRepository, OrderRepository>();
        services.AddScoped<IInvoiceRepository, InvoiceRepository>();
        services.AddScoped<OrderService>();
        services.AddScoped<InvoiceService>();
        services.AddSingleton<RetryHelper>();
        return services;
    }
}
