namespace Shop.Billing;

public sealed class RetryHelper
{
    private readonly ILogger logger;

    public RetryHelper(ILogger logger) { this.logger = logger; }

    public async Task<TResult> ExecuteWithRetryAsync<TResult>(Func<Task<TResult>> operation, int maxAttempts = 5)
    {
        var wait = TimeSpan.FromMilliseconds(500);
        for (var n = 1; ; n++)
        {
            try
            {
                return await operation();
            }
            catch (HttpRequestException error) when (n < maxAttempts && !error.Message.Contains("404"))
            {
                this.logger.Warn($"Try {n} failed: {error.Message}. Waiting {wait.TotalMilliseconds}ms");
                await Task.Delay(wait);
                wait = TimeSpan.FromMilliseconds(wait.TotalMilliseconds * 2);
            }
        }
    }
}
