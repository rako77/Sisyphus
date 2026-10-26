namespace Shop.Shared;

public class Customer : IEquatable<Customer>
{
    public Customer(string id, string name, string country)
    {
        Id = id;
        Name = name;
        Country = country;
    }

    public string Id { get; }
    public string Name { get; }
    public string Country { get; }
    public bool IsVip { get; set; }
    public int YearsActive { get; set; }

    public bool Equals(Customer? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        return Id == other.Id && Name == other.Name && Country == other.Country;
    }

    public override bool Equals(object? obj) => obj is Customer c && Equals(c);

    public override int GetHashCode()
    {
        unchecked
        {
            var hash = 17;
            hash = hash * 31 + Id.GetHashCode();
            hash = hash * 31 + Name.GetHashCode();
            hash = hash * 31 + Country.GetHashCode();
            return hash;
        }
    }
}

public class Address : IEquatable<Address>
{
    public Address(string street, string city, string country)
    {
        Street = street;
        City = city;
        Country = country;
    }

    public string Street { get; }
    public string City { get; }
    public string Country { get; }

    public bool Equals(Address? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        return Street == other.Street && City == other.City && Country == other.Country;
    }

    public override int GetHashCode()
    {
        unchecked
        {
            var hash = 17;
            hash = hash * 31 + Street.GetHashCode();
            hash = hash * 31 + City.GetHashCode();
            hash = hash * 31 + Country.GetHashCode();
            return hash;
        }
    }
}
