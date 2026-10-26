var builder = WebApplication.CreateBuilder(args);
builder.Services.AddShop();
var app = builder.Build();
app.Run();
